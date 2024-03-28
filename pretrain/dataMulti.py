import gc
import os
import sys
# sys.path.append("../")
import torch
from torch.utils.data import SequentialSampler, DataLoader, TensorDataset
from tqdm import tqdm

from MainModel.BertNlpModel import BertNlpModel
from tool.pytorch_pretrained_bert.modeling import BertForMaskedLM, BertConfig

import model_args as args

from tool.pytorch_pretrained_bert.tokenization import BertTokenizer
from tool.utils.ToolUtils import ToolUtils
from tool.utils.ChunkSave import ChunkSave
from tool.utils.ChunkArgs import ChunkArgs
import pandas as pd
import numpy as np

psedo_label_dir = "psedo"


def create_label(epoch):
    pretrain = BertForMaskedLM(config=BertConfig.from_json_file(args.bert_config_json))
    trainModel = BertNlpModel(pretrain)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    trainModel.to(device)
    model_weights = torch.load('./' + str(args.train_dir) + "/" + str(epoch) + args.save_name + ".bin",
                               map_location=device)
    trainModel.load_state_dict(model_weights)

    bTokenizer = BertTokenizer(vocab_file=args.vocab_file)
    predict_data = getPredictData("../data", "test_a.csv", bTokenizer)
    print(len(predict_data))
    predict_sampler = SequentialSampler(predict_data)

    test_dataloader = DataLoader(predict_data, sampler=predict_sampler, batch_size=args.eval_batch_size, num_workers=1)

    trainModel.eval()
    predicts_list = []
    input_ids_list = []
    input_masks_list = []
    segment_ids_list = []

    cargs = ChunkArgs(".", psedo_label_dir, 1)
    chunkSave = ChunkSave(cargs)

    for step, batch in enumerate(tqdm(test_dataloader, desc=" eval Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids = batch
        output = trainModel(input_ids, segment_ids, input_mask)
        output = torch.softmax(output, dim=-1)
        predict_label = torch.argmax(output, dim=-1, keepdim=False)
        # print(output.tolist())
        # print(output)
        _, selected_index = ToolUtils.getPseudoIndex(np.array(output.tolist()), ver=0.8)
        input_id_list = np.array(input_ids.tolist())[selected_index]
        # print(np.array(input_ids.tolist()).shape, selected_index.shape,input_id_list)
        input_mask_list = list(np.array(input_mask.tolist())[selected_index])
        segment_id_list = list(np.array(segment_ids.tolist())[selected_index])
        predict_label_list = list(np.array(predict_label.tolist())[selected_index])
        #print(len(input_mask_list))

        input_ids_list.extend(input_id_list)
        input_masks_list.extend(input_mask_list)
        segment_ids_list.extend(segment_id_list)
        predicts_list.extend(predict_label_list)
        #print("---", len(input_ids_list))
        #if step == 2:
        #    break

    print(len(predicts_list), len(input_id_list))
    data_list = []
    for i in range(len(input_ids_list)):
        data = {
            "input_ids": input_ids_list[i],
            "input_masks": input_masks_list[i],
            "segment_ids": segment_ids_list[i],
            "label": predicts_list[i]
        }
        data_list.append(data)
    chunkSave.save(data_list)

    # dataframe = pd.DataFrame(data=[[i] for i in predicts_list],columns=["label"])
    # dataframe.to_csv("res.csv",index=None)


def get_psedo_data():
    gc.disable()
    cargs = ChunkArgs(".", psedo_label_dir, 1)
    chunkSave = ChunkSave(cargs)
    chunkSave.load()
    data = chunkSave.getdata()
    input_id_list = []
    input_mask_list = []
    segment_id_list = []
    psedo_label_list = []

    for i in range(len(data)):
        input_id = data[i]["input_ids"]
        input_mask = data[i]["input_masks"]
        segment_id = data[i]["segment_ids"]
        psedo_label = data[i]["label"]

        input_id_list.append(input_id)
        input_mask_list.append(input_mask)
        segment_id_list.append(segment_id)
        psedo_label_list.append(psedo_label)

    return input_id_list, input_mask_list, segment_id_list, psedo_label_list


"""
    getData(path,name)
   获取训练数据,
   path路径,
   name表示名字
"""


def getTrainData(path, name, tokenizer):
    data = pd.read_csv(os.path.join(path + "/", name), sep="\t")
    # print(data)
    # print(data['text'])
    label_ids = []
    input_ids = []
    segment_ids = []
    input_masks = []
    # print(len(data))

    # 读取数据，转换输入为id
    for i in range(len(data['label'])):
        inputText = data['text'][i].split(" ")
        if len(inputText) > args.max_seq_length - 2:
            inputText = inputText[:args.max_seq_length - 2]

        inputText = ["[CLS]"] + inputText + ["[SEP]"]

        input_id = tokenizer.convert_tokens_to_ids(inputText)
        segment_id = [0 for _ in range(len(input_id))]
        label_id = data['label'][i]

        # 对齐数据长度
        assert len(input_id) == len(segment_id) <= args.max_seq_length

        input_id_array = np.zeros(args.max_seq_length, dtype=np.int_)
        input_id_array[:len(input_id)] = input_id

        segment_array = np.zeros(args.max_seq_length, dtype=np.bool_)
        segment_array[:len(input_id)] = segment_id

        input_mask_array = np.zeros(args.max_seq_length, dtype=np.bool_)
        input_mask_array[:len(input_ids)] = 1

        segment_ids.append(segment_array)
        input_ids.append(input_id_array)
        label_ids.append(label_id)
        input_masks.append(input_mask_array)
    print(np.array(input_masks).shape)
    if args.is_use_psdo:
        psdo_input_id_list, psdo_input_mask_list, \
        psdo_segment_id_list, psedo_label_list = get_psedo_data()

        input_ids.extend(psdo_input_id_list)
        input_masks.extend(psdo_input_mask_list)
        segment_ids.extend(psdo_segment_id_list)
        label_ids.extend(psedo_label_list)

    print(np.array(input_masks).shape)

    ## 训练和验证集二八分
    from sklearn.model_selection import StratifiedKFold

    import random
    random.seed(1)
    np.random.seed(1)
    kf = StratifiedKFold(n_splits=args.total_fold_num)

    input_ids_array = np.array([f for f in input_ids])
    segment_ids_array = np.array([f for f in segment_ids])
    label_ids_array = np.array([f for f in label_ids])
    input_masks_array = np.array([f for f in input_masks])

    # group_dict = ToolUtils.countGroup(label_ids_array)
    # print("total train", group_dict)
    # weight = [0] * len(group_dict.keys())
    # for i in group_dict.keys():
    #     weight[i] = 38917/group_dict[i]
    # print("weight",weight)
    # weight_tensor = torch.softmax(torch.tensor(np.array(weight), dtype=torch.float),dim=-1)
    # print(weight_tensor)

    kf.get_n_splits(input_ids_array, label_ids_array)
    step = 1
    for train_index, val_index in kf.split(input_ids_array, label_ids_array):
        train_ids, val_ids = input_ids_array[train_index], input_ids_array[val_index]
        train_segment_ids, val_segment_ids = segment_ids_array[train_index], segment_ids_array[val_index]
        train_label_ids, val_label_ids = label_ids_array[train_index], label_ids_array[val_index]
        print("train", ToolUtils.countGroup(train_label_ids))
        print("val", ToolUtils.countGroup(val_label_ids))
        train_input_masks, val_input_masks = input_masks_array[train_index], input_masks_array[val_index]
        if step == args.current_kfold_num:
            print("KFlold", step)
            break
        step = step + 1

    all_train_input_ids = torch.tensor(train_ids, dtype=torch.long)
    all_train_segment_ids = torch.tensor(train_segment_ids, dtype=torch.long)
    all_train_label_ids = torch.tensor(train_label_ids, dtype=torch.long)
    all_train_input_masks = torch.tensor(train_input_masks, dtype=torch.long)

    all_val_input_ids = torch.tensor(val_ids, dtype=torch.long)
    all_val_segment_ids = torch.tensor(val_segment_ids, dtype=torch.long)
    all_val_label_ids = torch.tensor(val_label_ids, dtype=torch.long)
    all_val_input_masks = torch.tensor(val_input_masks, dtype=torch.long)

    print(all_train_input_ids.size(), all_train_segment_ids.size(), all_train_label_ids.size(),
          all_train_input_masks.size())
    print(all_val_input_ids.size(), all_val_segment_ids.size(), all_val_label_ids.size(),
          all_val_input_masks.size())
    print("total label", np.unique(all_train_label_ids), " ", np.unique(all_val_label_ids))

    train_data = TensorDataset(all_train_input_ids, all_train_input_masks, all_train_segment_ids, all_train_label_ids)
    val_data = TensorDataset(all_val_input_ids, all_val_input_masks, all_val_segment_ids, all_val_label_ids)

    # del all_train_input_ids,all_train_segment_ids,all_train_label_ids,all_train_input_masks
    # del all_val_input_ids,all_val_segment_ids,all_val_label_ids,all_val_input_masks
    # del input_ids_array,segment_ids_array,label_ids_array,input_masks_array

    return train_data, val_data


def getPredictData(path, name, tokenizer):
    data = pd.read_csv(os.path.join(path + "/", name), sep="\t")
    # print(data['text'])
    input_ids = []
    segment_ids = []
    input_masks = []

    # 读取数据，转换输入为id
    for i in range(len(data['text'])):
        inputText = data['text'][i].split(" ")
        if len(inputText) > args.max_seq_length - 2:
            inputText = inputText[:args.max_seq_length - 2]

        inputText = ["[CLS]"] + inputText + ["[SEP]"]

        input_id = tokenizer.convert_tokens_to_ids(inputText)
        segment_id = [0 for _ in range(len(input_id))]

        # 对齐数据长度
        assert len(input_id) == len(segment_id) <= args.max_seq_length

        input_id_array = np.zeros(args.max_seq_length, dtype=np.int_)
        input_id_array[:len(input_id)] = input_id

        segment_array = np.zeros(args.max_seq_length, dtype=np.bool_)
        segment_array[:len(input_id)] = segment_id

        input_mask_array = np.zeros(args.max_seq_length, dtype=np.bool_)
        input_mask_array[:len(input_ids)] = 1

        segment_ids.append(segment_array)
        input_ids.append(input_id_array)
        input_masks.append(input_mask_array)

    input_ids_array = torch.tensor(np.array([f for f in input_ids]), dtype=torch.long)
    segment_ids_array = torch.tensor(np.array([f for f in segment_ids]), dtype=torch.long)
    input_masks_array = torch.tensor(np.array([f for f in input_masks]), dtype=torch.long)

    # print(input_ids_array.size(), segment_ids_array.size(), input_masks_array.size())
    test_data = TensorDataset(input_ids_array, input_masks_array, segment_ids_array)
    return test_data


if __name__ == "__main__":
    create_label(4)

