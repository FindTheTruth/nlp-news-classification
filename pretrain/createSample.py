from __future__ import absolute_import, division, print_function

import pretraining_args as args
import csv
import logging
import os
import random

from tool.utils.ToolUtils import ToolUtils
from tool.utils.ChunkArgs import ChunkArgs
from tool.utils.ChunkSave import ChunkSave
# args.seed
# random.seed(args.seed)
random.seed(35)

from random import random, randrange, randint, shuffle, choice, sample


import gc
import time

logger = logging.getLogger(__name__)


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


def create_examples(data_path, max_seq_length, masked_lm_prob, max_predictions_per_seq, vocab_list, name="train",epoch=10):
    """Creates examples for the training and dev sets."""
    examples = []
    max_num_tokens = max_seq_length - 2

    count = ToolUtils.getFileline(data_path)
    print("total data count ", count)
    current_chunk = 0
    for k in range(epoch):
        cargs = ChunkArgs(args.pretrain_data_dir, name + str(k), args.chunk_num)
        cargs.setProxyConfig(1000000, count)
        chunkSave = ChunkSave(cargs)
        print("epoch ", str(k),"starting process")
        fr = open(data_path, "r")
        for (i, line) in enumerate(fr):
            gc.disable()
            current_timestamp = time.time()
            if chunkSave.proxySavePart(i, examples):
                ttime = time.time()
                time_cost = ttime - current_timestamp
                print("time cost ", time_cost)
                del examples
                examples = []
                old_dir = os.path.join(args.pretrain_data_dir +"/",name + str(k) + str(current_chunk%args.chunk_num))+".pkl"
                new_dir = os.path.join(args.pretrain_data_dir +"/",name+str(current_chunk)+".pkl")
                print("from old dir:",old_dir,"rename to",new_dir)
                if os.path.exists(new_dir):
                    print("old file exist,delete",new_dir)
                    os.remove(new_dir)
                os.rename(old_dir,new_dir)
                current_chunk = current_chunk + 1
            whole_text = line.strip("\n").split()
            tokens_a = whole_text[:128]
            tokens_a.extend(whole_text[len(whole_text)-(max_num_tokens-128):])
            # if len(tokens_a) >= 510:
            #     print(len(tokens_a),tokens_a)
            # tokens_a = line.strip("\n").split()[:max_num_tokens]
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
    del examples

def main():
    vocab_list = []
    with open(args.vocab_file, 'r') as fr:
        for line in fr:
            vocab_list.append(line.strip("\n"))

    if args.do_train:
        create_examples(data_path=args.pretrain_train_path,
                        max_seq_length=args.max_seq_length,
                        masked_lm_prob=args.masked_lm_prob,
                        max_predictions_per_seq=args.max_predictions_per_seq,
                        vocab_list=vocab_list)


if __name__ == "__main__":
    main()