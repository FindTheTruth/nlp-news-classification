import warnings

import numpy as np


class ToolUtils:
    """
    获取文件的行数
    fr = open(data_path, "r")
    line = getFileLine(fr)
    """

    @staticmethod
    def getFileline(data_path):
        fr = open(data_path, "r")
        lineCount = sum(1 for line in fr)
        return lineCount

    """
        获取内存的使用量
    """

    @staticmethod
    def getMemoryCount():
        import psutil
        # 获取当前进程对象
        process = psutil.Process()
        # 获取当前进程的内存使用情况
        memory_info = process.memory_full_info().rss / (1024 * 1024)
        return memory_info

    """
       data->List 输入标签[0,1,2,3,1,2]
       返回一个dict，对应每个标签的数目
    """

    @staticmethod
    def countGroup(data):
        count_key = {}
        for i in data:
            if i in count_key.keys():
                count_key[i] = count_key[i] + 1
            else:
                count_key[i] = 0
        return dict(sorted(count_key.items(), key=lambda x: x[1]))

    """
        
        输入array：label [0,1,1,0] predict[1,0,1,0]
        返回f1 score
    """

    @staticmethod
    def getF1Score(label, predict):
        from sklearn.metrics import f1_score
        res = f1_score(label, predict, average="macro")
        return res

    """
           array:
           输入：array label [0,1,1,0] predict[1,0,1,0]
           返回不同label的f1 score,
           Dict格式，key为label,value为f1 score
    """

    @staticmethod
    def getF1ScoreByGroup(label, predict):
        from sklearn.metrics import f1_score
        unique_label = np.unique(label)
        f1_dict = {}
        for i in unique_label:
            # i标签的真实数目
            label_sum = np.sum(np.array(label == i))
            # i标签的预测数目
            predict_i = np.sum(np.array(predict == i))
            # i标签真实值中预测对的数目
            predict_true = np.sum(np.array(predict[np.array(label == i)]) == i)
            # i标签预测值中，真实猜中的数目
            true_count = np.sum(np.array(label[np.array(predict == i)] == i))
            # print(label_sum,predict_true,true_count,predict_i)
            precision = (predict_true + 1e-5) / (label_sum + 1e-5)
            recall = (true_count + 1e-5) / (predict_i + 1e-5)
            print("label", i, " precision", precision, "recall", recall)
            f1 = 2 * (precision * recall) / (precision + recall)
            f1_dict[i] = f1

        return f1_dict

    """
              array:
              输入：array prob_label [[0.2,0.3,0.5],[0.4,0.2,0.4],[0.3,0.3,0.4]] 
                   ver: 阈值0.5
              1.返回prob_label中预测最大值 大于阈值的index
              2.array格式 按下标大小排序返回对应数据集的下标，true,false
              3.返回对应数据的预测label
              
              举例：
                array prob_label [[0.2,0.3,0.5],[0.4,0.2,0.4],[0.3,0.3,0.4]] 
                ver: 阈值0.5
              返回：array[0,2] array[[True],[False],[True]],
              
    for step, batch in enumerate(tqdm(test_dataloader, desc=" eval Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids = batch
        output = trainModel(input_ids, segment_ids, input_mask)
        output = torch.softmax(output, dim=-1)
        predict_label = torch.argmax(output, dim=-1, keepdim=False)

        _,selected_index = ToolUtils.getPseudoIndex(np.array(output.tolist()),ver=0.5)
        input_id_list = list(np.array(input_ids.tolist())[selected_index])
        input_mask_list =list(np.array(input_mask.tolist())[selected_index])
        segment_ids_list =list(np.array(segment_ids.tolist())[selected_index])
       """

    @staticmethod
    def getPseudoIndex(label, ver=0.4):
        # 生成[0,1,2,3,4....]下标
        indexs_array = np.array([i for i in range(label.shape[0])])
        predict_label = np.argmax(label, axis=1,keepdims=False)
        # 筛选符合条件的label
        index = np.array(np.amax(label, axis=1)) >= ver

        return indexs_array[index], index, predict_label

    """
       input: result:array[[1,2],[1,0],[1,0]]
              每个元素都是一个模型预测的结果
              class_num:总共的labal数目[从0开始计数]
              n:整型数字，用于筛选针对某个样本，预测器有大于n结果的数据
       output: res(array):返回投票的模型预测结果。如果所有模型的预测结果都不一致，随机选择其中的一个作为标签。比如例子返回[1,0]
               index:返回预测结果n个模型都不一致的index
               
        使用样例：
        label = np.array([[1, 2 ,1], [1, 0,1], [1, 0,1],[1,2,2],[1,2,0]])
        ToolUtils.mergeClassifyResult(label,3,2)
    """

    @staticmethod
    def mergeClassifyResult(result, class_num, n):
        class_predict = []
        # if len(np.unique(result[0]))!=class_num:
        #     warnings.warn("check class_num ?")
        # 计算每个样本对应类的预测数目
        for i in range(class_num):
            class_i_predict = np.sum(result == i, axis=0)
            class_predict.append(class_i_predict)
        class_predict = np.array(class_predict)
        # 预测数目和预测期的数量不一致报错
        if np.sum(np.sum(class_predict, axis=0) != len(result)) != 0:
            warnings.warn("check data")

        res = np.argmax(class_predict, 0)
        # 每个样本预测器预测出不一样值的个数，比如[[0,1],[1,1]],预测器预测的结果不一致,index返回为[2,1]
        class_count = np.sum(class_predict !=0, axis=0)

        index = [i for i,x in enumerate(list(class_count>n)) if x ==True]

        # print(class_predict)
        # print(res,index)
        return res,index

    """
        返回模型分类：TOP2概率差异小于ver的样本
        输入：
            array prob_label(输入值必须大于1，便于计算第二大值) [[0.2,0.3,0.5],[0.4,0.2,0.4],[0.3,0.3,0.4]]
            ver = 0.1
        输出：
            返回list:
            1.对应样本的index
            2.对应样本TOP2标签，长度为2，第一个list标明top1的label值，第二个表明top2的label
            2.对应样本概率值[list]第一个list标明top1的概率值，第二个表明top2的概率
        egg:
            input = np.array([[0.2,0.3,0.5],[0.4,0.2,0.4],[0.3,0.3,0.7]])
            print(ToolUtils.getTopKData(input))
            print(input)
            
            index, predict_labels, predict_probs = ToolUtils.getTopKData(np.array(predict_prob), ver=0.3)
            true_labels = np.array(label)[index]
            dataframe = pd.DataFrame({"true_label": list(true_labels),
                              "predict_label_1": list(np.array(predict_labels[0])[index]),
                              "predict_label_2": list(np.array(predict_labels[1])[index]),
                              "predict_label_prob1": list(np.array(predict_probs[0])[index]),
                              "predict_label_prob2": list(np.array(predict_probs[1])[index])})
            dataframe.to_csv("analysis.csv", index=None)

            
    """
    @staticmethod
    def getTopKData(prob_label, ver=0.1):
        indexs_array = np.array([i for i in range(prob_label.shape[0])])
        prob_label = prob_label.copy()
        predict_first_label = np.argmax(prob_label, axis=1, keepdims=False)
        predict_first_prob = np.max(prob_label,axis=1, keepdims=False)
        # 筛选符合条件的label
        # first_index = np.array(np.amax(prob_label, axis=1)) >= ver
        # print(predict_first_label, predict_first_prob)

        # 第一大的概率预测修改为-1
        for i in range(prob_label.shape[0]):
            prob_label[i][predict_first_label[i]] = -1

        predict_second_label = np.argmax(prob_label, axis=1, keepdims=False)
        predict_second_prob = np.max(prob_label, axis=1, keepdims=False)
        # print(predict_second_label,predict_second_prob)
        # print(predict_first_prob - predict_second_prob < ver)
        index = indexs_array[predict_first_prob - predict_second_prob < ver]

        predict_labels = [list(predict_first_label), list(predict_second_label)]

        predict_probs = [list(predict_first_prob), list(predict_second_prob)]

        return index, predict_labels, predict_probs


# input = np.array([[0.2,0.3,0.5],[0.4,0.2,0.4],[0.3,0.3,0.7]])
# print(ToolUtils.getTopKData(input))
# print(input)


