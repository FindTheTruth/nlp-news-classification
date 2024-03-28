import logging
import os
import sys

# from tool.utils.ChunkArgs import ChunkArgs
# from tool.utils.ChunkSave import ChunkSave
from tool.utils.EMA import EMA

sys.path.append("../")
import numpy as np
import pandas as pd
import torch
import gc
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from tool.utils.ToolUtils import ToolUtils
from tqdm import trange, tqdm
from transformers import get_linear_schedule_with_warmup, AdamW

from BertNlpModel import BertNlpModel
from MainModel.TrainOptimizer import MyOptimizer

from tool.pytorch_pretrained_bert.modeling import BertForMaskedLM, BertConfig
import model_args as args
from tool.pytorch_pretrained_bert.tokenization import BertTokenizer
from tool.pytorch_pretrained_bert.optimization import BertAdam
from dataMulti import getTrainData, getPredictData

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

filename = "./" + str(args.train_dir) + "/" + args.log_name
#
file_handler = logging.FileHandler(filename)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def loadModel(path, name):
    modelPath = os.path.join(path + "/", name + ".bin")
    optimierPath = os.path.join(path + "/", name + "op.bin")
    model = BertForMaskedLM(config=BertConfig.from_json_file(args.bert_config_json))
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(params=optimizer_grouped_parameters, lr=args.learning_rate,
                         warmup=args.warmup_proportion)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    model_weights = torch.load(modelPath, map_location=device)
    model.load_state_dict(model_weights)

    optimizer_weights = torch.load(optimierPath, map_location="cpu")
    optimizer.load_state_dict(optimizer_weights)

    # print(model_weights.keys())
    # print("\n\n\n")
    # print(model.state_dict().keys())
    #
    # print(optimierPath)

    # print(optimizer_weights)
    # print("\n\n\n")
    # print(optimizer.state_dict()['param_groups'][0]['schedule'].warmup)
    # print(optimizer.state_dict()['param_groups'][0]['schedule'].get_lr_(0.2))
    return model, optimizer


def MyOptimizerSetting(model, steps):
    myTrainOptimizer = MyOptimizer(model.all_parameters, steps)

    return myTrainOptimizer


def loggingSetting(filename):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    file_handler = logging.FileHandler(filename)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)


def start_trainer():
    model, _ = loadModel("./pretrainModel", "10")

    trainModel = BertNlpModel(model)

    bTokenizer = BertTokenizer(vocab_file=args.vocab_file)
    train_data, val_data = getTrainData("../data", "train_set.csv", bTokenizer)
    print(len(train_data), len(val_data))
    train_sampler = RandomSampler(train_data)
    val_sampler = SequentialSampler(val_data)

    # save_path = './'+str(args.train_dir)
    # if os.path.exists(save_path) and os.listdir(save_path) and args.do_train:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)

    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=1)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=args.eval_batch_size, num_workers=1)

    steps = int(args.training_epoch * len(train_data) / args.train_batch_size)

    optimizer = MyOptimizerSetting(trainModel, steps)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    trainModel.to(device)
    epoch = 0
    num_class = 14
    weighted_list = [1.0 / 14] * num_class
    print("init", weighted_list)
    # weighted = torch.nn.Softmax(torch.tensor(np.array(weighted_list, dtype=float), dtype=torch.float))
    weighted = torch.tensor(np.array(weighted_list, dtype=float), dtype=torch.float).to(device)
    # ema = EMA(trainModel, 0.999)
    # ema.register()
    for e in trange(int(args.training_epoch), desc="Epoch"):
        trainModel.train()
        epoch += 1
        tr_loss = 0
        nb_tr_steps = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            # masked_lm_loss
            output = trainModel(input_ids, segment_ids, input_mask)
            loss_fct = CrossEntropyLoss()
            # loss_fct = CrossEntropyLoss(weighted)
            # print("output size ",output.size())
            loss = loss_fct(output, label_ids)
            # print("step:loss ", loss.item())
            loss.backward()

            tr_loss += loss.item()
            nb_tr_steps += 1

            optimizer.step()
            optimizer.zero_grad()

            if nb_tr_steps > 0 and nb_tr_steps % 100 == 0:
                logger.info("===================== -epoch %d -train_step %d -train_loss %.4f\n" % (
                    e, nb_tr_steps, tr_loss / nb_tr_steps))
            # ema.update()
            break

        trainModel.eval()
        predict = []
        label = []
        # ema.apply_shadow()
        for step, batch in enumerate(tqdm(val_dataloader, desc=" val Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            output = trainModel(input_ids, segment_ids, input_mask)
            output = torch.argmax(torch.softmax(output, dim=-1), dim=-1, keepdim=False)
            predict.extend(output.tolist())
            label.extend(label_ids.tolist())
            # break

        from sklearn.metrics import f1_score
        res = f1_score(label, predict, average="macro")
        logger.info("res:  val %s", res)
        my_dict = ToolUtils.getF1ScoreByGroup(np.array(label), np.array(predict))
        logger.info("res is %s", my_dict.items())

        max_value = max(my_dict.values())
        print("max f1:", max_value)
        for key, values in my_dict.items():
            weighted_list[key] = max_value / my_dict[key]
        logger.info("weight list %s", weighted_list)
        del weighted
        weighted = torch.tensor(np.array(weighted_list, dtype=float), dtype=torch.float).to(device)
        weighted = torch.softmax(weighted, dim=-1)
        logger.info("weighted %s", weighted.tolist())

        # print(ToolUtils.getF1ScoreByGroup(np.array(label), np.array(predict)))

        model_to_save = trainModel.module if hasattr(trainModel,
                                                     'module') else trainModel  # Only save the model it-self
        # # # If we save using the predefined names, we can load using `from_pretrained`
        # output_model_file = os.path.join('./'+str(args.train_dir), str(epoch) + args.save_name+".emabin")
        # # output_model_optimizer = os.path.join('./outputs', str(epoch) + '.0-2opbin')
        # torch.save(model_to_save.state_dict(), output_model_file)
        # # torch.save(optimizer.state_dict(), output_model_optimizer)
        # ema.restore()
        output_model_file = os.path.join('./' + str(args.train_dir), str(epoch) + args.save_name + ".bin")
        # output_model_optimizer = os.path.join('./outputs', str(epoch) + '.0-2opbin')
        torch.save(model_to_save.state_dict(), output_model_file)


def analysis_data(epoch):
    pretrain = BertForMaskedLM(config=BertConfig.from_json_file(args.bert_config_json))
    trainModel = BertNlpModel(pretrain)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    loadpath = './' + str(args.train_dir) + "/" + str(epoch) + args.save_name + ".bin"
    print(loadpath)
    model_weights = torch.load(loadpath,
                               map_location=device)
    trainModel.load_state_dict(model_weights)

    bTokenizer = BertTokenizer(vocab_file=args.vocab_file)
    train_data, val_data = getTrainData("../data", "train_set.csv", bTokenizer)
    del train_data
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=args.eval_batch_size)

    trainModel.to(device)
    trainModel.eval()
    predict = []
    predict_prob = []
    label = []
    for step, batch in enumerate(tqdm(val_dataloader, desc=" val Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        output = trainModel(input_ids, segment_ids, input_mask)
        output_prob = torch.softmax(output, dim=-1)
        output = torch.argmax(output_prob, dim=-1, keepdim=False)
        predict_prob.extend(output_prob.tolist())
        predict.extend(output.tolist())
        label.extend(label_ids.tolist())

    from sklearn.metrics import f1_score
    res = f1_score(label, predict, average="macro")
    logger.info("res:  val %s", res)
    print(ToolUtils.getF1ScoreByGroup(np.array(label), np.array(predict)))
    # ToolUtils.getPseudoIndex(1-np.array(output_prob),0.5)
    index, predict_labels, predict_probs = ToolUtils.getTopKData(np.array(predict_prob), ver=0.3)
    true_labels = np.array(label)[index]

    dataframe = pd.DataFrame({"true_label": list(true_labels),
                              "predict_label_1": list(predict_labels[0]),
                              "predict_label_2": list(predict_labels[1]),
                              "predict_label_prob1": list(predict_probs[0]),
                              "predict_label_prob2": list(predict_probs[1])})
    dataframe.to_csv("analysis.csv", index=None)


def start_predict(epoch):
    pretrain = BertForMaskedLM(config=BertConfig.from_json_file(args.bert_config_json))
    trainModel = BertNlpModel(pretrain)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    trainModel.to(device)
    model_weights = torch.load('./' + str(args.train_dir) + "/" + str(epoch) + args.save_name + ".bin",
                               map_location=device)
    trainModel.load_state_dict(model_weights)

    bTokenizer = BertTokenizer(vocab_file=args.vocab_file)
    predict_data = getPredictData("../data", "test_a.csv", bTokenizer)
    predict_sampler = SequentialSampler(predict_data)

    test_dataloader = DataLoader(predict_data, sampler=predict_sampler, batch_size=args.eval_batch_size, num_workers=1)

    trainModel.eval()
    predicts = []
    pre_examples = []
    gc.disable()
    for step, batch in enumerate(tqdm(test_dataloader, desc=" eval Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids = batch
        output = trainModel(input_ids, segment_ids, input_mask)
        output_prob = torch.softmax(output, dim=-1)
        output = torch.argmax(output_prob, dim=-1, keepdim=False)

        # 筛选满足条件的样例

        predicts.extend(output.tolist())
        # break

    dataframe = pd.DataFrame(data=[[i] for i in predicts], columns=["label"])
    dataframe.to_csv("res.csv", index=None)


if __name__ == "__main__":
    # loggingSetting("./" +str(args.train_dir) + "/" + args.log_name)
    # analysis_data(4)
    if not args.is_val:
        start_trainer()
    else:
        start_predict(4)
