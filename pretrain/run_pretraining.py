from __future__ import absolute_import, division, print_function

import pretraining_args as args
import csv
import logging
import os
import random
import sys
sys.path.append("../")
os.path.join("../")
from tool.utils.ToolUtils import ToolUtils
from tool.utils.ChunkArgs import ChunkArgs
from tool.utils.ChunkSave import ChunkSave

random.seed(args.seed)
import sys
from glob import glob
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from random import random, randrange, randint, shuffle, choice, sample
from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from tool.pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from tool.pytorch_pretrained_bert.modeling import BertForMaskedLM, BertConfig
from tool.pytorch_pretrained_bert.tokenization import BertTokenizer
from tool.pytorch_pretrained_bert.optimization import BertAdam
import gc
import time

if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

logging.basicConfig(filename= args.output_dir + '/pretrain.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.DEBUG, filemode='a', datefmt='%Y-%m-%d%I:%M:%S %p')


# logger = logging.getLogger(__name__)
# file_handler = logging.FileHandler('log0-2.txt')
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indices.append(i)

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob))))
    # print(num_to_mask)
    # print("tokens", len(tokens))
    # print("cand", len(cand_indices))
    shuffle(cand_indices)
    mask_indices = sorted(sample(cand_indices, num_to_mask))
    masked_token_labels = []
    for index in mask_indices:
        # 80% of the time, replace with [MASK]
        if random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = choice(vocab_list)
        masked_token_labels.append(tokens[index])
        # Once we've saved the true label for that token, we can overwrite it with the masked version
        tokens[index] = masked_token

    return tokens, mask_indices, masked_token_labels


def create_examples(data_path, max_seq_length, masked_lm_prob, max_predictions_per_seq, vocab_list):
    """Creates examples for the training and dev sets."""
    examples = []
    max_num_tokens = max_seq_length - 2
    fr = open(data_path, "r")
    current_timestamp = time.time()

    # for (i, line) in tqdm(enumerate(fr), desc="Creating Example"):
    # chunkSave = ChunkSave(".", "pretraining", 5)
    count = ToolUtils.getFileline(data_path)
    print("total data count ", count)
    cargs = ChunkArgs(".", args.pretrain_data_dir, 5)
    cargs.setProxyConfig(1000000, count)
    chunkSave = ChunkSave(cargs)
    for (i, line) in enumerate(fr):
        gc.disable()
        # chunkSave.proxy_save(i,examples)
        current_timestamp = time.time()
        if chunkSave.proxySavePart(i, examples):
            ttime = time.time()
            time_cost = ttime - current_timestamp
            print("time cost ", time_cost)
            del examples
            examples = []

        tokens_a = line.strip("\n").split()[:max_num_tokens]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0 for _ in range(len(tokens_a) + 2)]
        # remove too short sample
        if len(tokens_a) < 5:
            continue
        tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
            tokens, masked_lm_prob, max_predictions_per_seq, vocab_list)
        example = {
            "tokens": tokens,
            "segment_ids": segment_ids,
            "masked_lm_positions": masked_lm_positions,
            "masked_lm_labels": masked_lm_labels}
        examples.append(example)
        gc.enable()
    fr.close()
    return examples


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    features = []
    current_time = time.time()
    for i, example in enumerate(examples):
        gc.disable()
        start_time = current_time
        if i != 0 and i % 1000000 == 0:
            end_time = time.time()
            print("cost", end_time - start_time)
            current_time = end_time
        tokens = example["tokens"]
        segment_ids = example["segment_ids"]
        masked_lm_positions = example["masked_lm_positions"]
        masked_lm_labels = example["masked_lm_labels"]
        assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

        input_array = np.zeros(max_seq_length, dtype=np.int_)
        input_array[:len(input_ids)] = input_ids

        mask_array = np.zeros(max_seq_length, dtype=np.bool_)
        mask_array[:len(input_ids)] = 1

        segment_array = np.zeros(max_seq_length, dtype=np.bool_)
        segment_array[:len(segment_ids)] = segment_ids

        lm_label_array = np.full(max_seq_length, dtype=np.int_, fill_value=-1)
        lm_label_array[masked_lm_positions] = masked_label_ids

        feature = InputFeatures(input_ids=input_array,
                                input_mask=mask_array,
                                segment_ids=segment_array,
                                label_id=lm_label_array)
        features.append(feature)
        gc.enable()

    return features


"""

获取当前加载数据块的数据对应的dataloader
start->end表示加载的块范围

"""


def getTrainDataLoader(chunkSave, start, end, tokenizer):
    logging.info("get dataloader start")
    chunkSave.load_range(start, end)
    gc.disable()
    pretrain_examples = chunkSave.getdata()
    logging.info("len data %d-%d,%d", start,end-1, len(pretrain_examples))
    train_features = convert_examples_to_features(
        pretrain_examples, args.max_seq_length, tokenizer)

    current_memory = ToolUtils.getMemoryCount()
    del pretrain_examples
    del chunkSave.data
    chunkSave.data = []
    del_memory = ToolUtils.getMemoryCount()

    logging.info("release memory %s M", current_memory - del_memory)

    logging.info("current memory %s M", ToolUtils.getMemoryCount())

    all_input_ids = torch.tensor(np.array([f.input_ids for f in train_features]), dtype=torch.long)
    logging.info("current memory %s M", ToolUtils.getMemoryCount())

    all_input_mask = torch.tensor(np.array([f.input_mask for f in train_features]), dtype=torch.long)
    logging.info("current memory %s M", ToolUtils.getMemoryCount())

    all_segment_ids = torch.tensor(np.array([f.segment_ids for f in train_features]), dtype=torch.long)
    logging.info("current memory %s M", ToolUtils.getMemoryCount())

    all_label_ids = torch.tensor(np.array([f.label_id for f in train_features]), dtype=torch.long)
    logging.info("current memory %s M", ToolUtils.getMemoryCount())

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    current_memory = ToolUtils.getMemoryCount()
    del train_features
    del all_input_ids, all_input_mask, all_segment_ids, all_label_ids
    del_memory = ToolUtils.getMemoryCount()
    logging.info("release tensor all memory %s M", current_memory - del_memory)

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    logging.info("get dataloader end")

    return train_dataloader


def main():
    print("main running")
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)

    tokenizer = BertTokenizer(vocab_file=args.vocab_file)

    train_examples = None
    num_train_optimization_steps = None
    vocab_list = []
    with open(args.vocab_file, 'r') as fr:
        for line in fr:
            vocab_list.append(line.strip("\n"))

    if args.do_train:
        gc.disable()
        cargs = ChunkArgs("./pretraining", "train", args.chunk_num*args.num_train_epochs)
        chunkSave = ChunkSave(cargs)
        # chunkSave.load()
        # train_examples = chunkSave.getdata()
        train_len = 4769137
        # print("train len:", len(train_examples))
        num_train_optimization_steps = int(
            train_len / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    model = BertForMaskedLM(config=BertConfig.from_json_file(args.bert_config_json))
    if args.fp16:
        model.half()

    model.to(device)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    best_loss = 100000

    if args.do_train:
        start = 0
        end = 0
        model.train()
        epoch = 0
        for e in trange(int(args.num_train_epochs * args.chunk_num/args.load_chunks), desc="Epoch"):
            epoch += 1
            tr_loss = 0

            start = end
            end = start + args.load_chunks

            nb_tr_examples, nb_tr_steps = 0, 0
            gc.disable()
            train_dataloader = getTrainDataLoader(chunkSave, start, end, tokenizer)
            gc.disable()

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                # masked_lm_loss
                loss = model(input_ids, segment_ids, input_mask, label_ids)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                if nb_tr_steps > 0 and nb_tr_steps % 100 == 0:
                    logging.info("===================== -epoch %d -train_step %d -train_loss %.4f\n" % (
                        e, nb_tr_steps, tr_loss / nb_tr_steps))
                #if nb_tr_steps % 100 == 0:
                #    break

            current_memory = ToolUtils.getMemoryCount()
            del train_dataloader
            del_memory = ToolUtils.getMemoryCount()
            logging.info(" data loader release memory %s M", current_memory - del_memory)

            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            # If we save using the predefined names, we can load using `from_pretrained`
            output_model_file = os.path.join(args.output_dir, str(epoch) + '.bin')
            output_model_optimizer = os.path.join(args.output_dir, str(epoch) + 'op.bin')
            torch.save(model_to_save.state_dict(), output_model_file)
            torch.save(optimizer.state_dict(), output_model_optimizer)
            # output_model_file = os.path.join('./outputs',)
            '''
            if True:
                eval_examples = create_examples(data_path=args.pretrain_dev_path,
                                         max_seq_length=args.max_seq_length,
                                         masked_lm_prob=args.masked_lm_prob,
                                         max_predictions_per_seq=args.max_predictions_per_seq,
                                         vocab_list=vocab_list)
                eval_features = convert_examples_to_features(
                    eval_examples, args.max_seq_length, tokenizer)
                all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
                eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
                # Run prediction for full data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval()
                eval_loss = 0
                nb_eval_steps = 0
                for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        loss = model(input_ids, segment_ids, input_mask, label_ids)

                    eval_loss += loss.item()
                    nb_eval_steps += 1

                eval_loss = eval_loss / nb_eval_steps
                if eval_loss < best_loss:
                    # Save a trained model, configuration and tokenizer
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

                    # If we save using the predefined names, we can load using `from_pretrained`
                    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    best_loss = eval_loss
                logger.info("============================ -epoch %d -train_loss %.4f -eval_loss %.4f\n"% (e, tr_loss / nb_tr_steps, eval_loss))
                '''


if __name__ == "__main__":
    main()
