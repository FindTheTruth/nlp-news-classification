import math
import pickle
import time
import os
import gc
# from tool.utils.ChunkArgs import ChunkArgs
import copy
from sys import getrefcount

"""

LIST 存储
data:表示具体的数据LIST格式
path:表示数据存储路径 egg:/opt/app
chunks:表示分块存储的数目
注意点：为了加快速度，采用了gc.disable()，加载结束后会启用gc.enable()
如果本身不需要python 自动回收机制，用完后记得调用gc.disable()
example:
    1.LIST为全集数据，分块存储，一并加载
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 23], [23, 45, 67], [89, 45, 23], [23, 45, 67], [78, 34, 21]]
    # 初始化，第一个参数表示目录，第二个表示文件名，第三个表示分的块数
    args = ChunkArgs(".","test",3)
    chunkSave = ChunkSave(args)
    # 存储为三块数据
    chunkSave.save(data)
    chunkSave.load()
    print(chunkSave.getdata())
    # 加载其中的一块数据，并获取
    chunkSave.load_part(0)
    print(chunkSave.getdata())
    
    2.LIST为数据中的一个chunk,可以单独存储，最后可以一并读取，加载到一个LIST中
   data1 = [[1, 2, 3], [4, 5, 6]]
    data2 = [[7, 8, 9], [10, 11, 23], [23, 45, 67]]
    data3 = [[89, 45, 23], [23, 45, 67], [78, 34, 21]]
    args = ChunkArgs(".","test",3)
    chunkSave = ChunkSave(args)
    # 分块存储
    chunkSave.save_part(data1,0)
    chunkSave.save_part(data2,1)
    chunkSave.save_part(data3,2)
    # 分块加载和读取
    chunkSave.load_part(1)
    print(chunkSave.getdata())
    # 一起读取
    chunkSave.load()
    print(chunkSave.getdata())

iter表示当前迭代的样本数,input表示当前的list
根据iter决定是否要存储input数据到对应的part内如果存储完成，应该有输出finish steps日志，没有需要检查
def proxySavePart(self, iter, input):
3.proxySavePart使用样例：

args = ChunkArgs(".", "pretraining", 5)
args.setProxyConfig(1000000,1000003)
examples = []
chunkSave = ChunkSave(args)
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
    example = {
            "tokens": tokens,
            "segment_ids": segment_ids,
            "masked_lm_positions": masked_lm_positions,
            "masked_lm_labels": masked_lm_labels}
    examples.append(example)
"""


class ChunkSave:
    def __init__(self, chunkargs):
        self.chunkargs = chunkargs
        self.data = []

    def save(self, data):
        length = len(data)
        self.data = data
        print("len:", length)
        chunk_len = math.ceil(length / self.chunkargs.chunks)
        print("total part", self.chunkargs.chunks)
        for i in range(self.chunkargs.chunks):
            if not i == self.chunkargs.chunks - 1:
                data_chunk = self.data[i * chunk_len:(i + 1) * chunk_len]
                print("\nprocessing part ", i, "from ", i * chunk_len, ":", (i + 1) * chunk_len)
            else:
                data_chunk = self.data[i * chunk_len:]
                print("\nprocessing part ", i, "from ", i * chunk_len, ":-1")
            print("start saving ")
            start_time = time.time()
            with open(self.chunkargs.path + "/" + self.chunkargs.name + str(i) + ".pkl", 'wb') as f:
                pickle.dump(data_chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
            end_time = time.time()
            print("cost:", end_time - start_time)

    """
        iter表示当前迭代的样本数,input表示当前的list
        根据iter决定是否要存储input数据到对应的part内
        如果存储完成，应该有输出finish steps日志，没有需要检查
        如果值得存储，就返回true
    """

    def proxySavePart(self, iter, input):
        # print(iter)
        if iter == 0:
            return False
        elif iter % self.chunkargs.save_steps == 0:
            print("saving steps:", iter, "chunk ", str(int(iter / self.chunkargs.save_steps) - 1))
            self.save_part(input, int(iter / self.chunkargs.save_steps) - 1)
            return True
        elif iter == self.chunkargs.data_len - 1:
            print("finish steps:", iter, "chunk ", str(int(iter / self.chunkargs.save_steps)))
            self.save_part(input, int(iter / self.chunkargs.save_steps))
            return True
        return False

    def save_part(self, partdata, num):
        self.data = partdata
        num = int(num)
        print("\nsaving ", num, "/", self.chunkargs.chunks - 1)
        start_time = time.time()
        if num >= self.chunkargs.chunks:
            raise Exception("NUM " + str(num) + "greater or equal than chunks" + str(self.chunkargs.chunks))
        with open(self.chunkargs.path + "/" + self.chunkargs.name + str(num) + ".pkl", 'wb') as f:
            pickle.dump(partdata, f, protocol=pickle.HIGHEST_PROTOCOL)
        end_time = time.time()
        print("save part num", num, "time cost", end_time - start_time, "\n")

    def load(self):
        print("\ntotal part", self.chunkargs.chunks)
        data_chunks = []
        # if self.chunkargs.data_len != 0:
        #     data_chunks = [0] * self.chunkargs.data_len
        del self.data
        self.data = []
        gc.disable()
        for i in range(self.chunkargs.chunks):
            data_chunk = []

            start_time = time.time()
            try:
                with open(self.chunkargs.path + "/" + self.chunkargs.name + str(i) + ".pkl", 'rb') as f:
                    data_chunk = pickle.load(f)
            except FileNotFoundError as e:
                print("excpetion,file not found ", str(i) + ".pkl")
            data_chunks.extend(data_chunk)
            end_time = time.time()
            print("chunk", str(i), " cost:", end_time - start_time)
        print("give input")
        self.data = data_chunks
        print("finish input")
        gc.enable()

    def load_part(self, num):
        gc.disable()
        start_time = time.time()
        with open(self.chunkargs.path + "/" + self.chunkargs.name + str(num) + ".pkl", 'rb') as f:
            data_chunk = pickle.load(f)
        end_time = time.time()
        print("chunk", str(num), " cost:", end_time - start_time)
        gc.enable()
        del self.data
        self.data = data_chunk
        return data_chunk

    """
    加载指定范围内的chunks,start->end之间的
    注意start,end必须在chunk范围内，chunk范围外报错
    """

    def load_range(self, start, end):
        if start > self.chunkargs.chunks or start < 0:
            print("load start", start, "failed,value is greater than", self.chunkargs.chunks)
        if end > self.chunkargs.chunks or start > end:
            print("load end", end, "failed,value is less than chunks or start is greater than end")
        print("\ntotal part", self.chunkargs.chunks)
        data_chunks = []
        del self.data
        self.data = []
        gc.disable()
        for i in range(start, end):
            data_chunk = []
            start_time = time.time()
            try:
                with open(self.chunkargs.path + "/" + self.chunkargs.name + str(i) + ".pkl", 'rb') as f:
                    data_chunk = pickle.load(f)
            except FileNotFoundError as e:
                print("excpetion,file not found ", self.chunkargs.path + "/" + self.chunkargs.name + str(i) + ".pkl")
            data_chunks.extend(data_chunk)
            end_time = time.time()
            print("chunk", str(i), " cost:", end_time - start_time)
        print("give input")
        self.data = data_chunks
        print("finish input")
        gc.enable()

    def getdata(self):
        print("get data")
        if self.data is None:
            raise Exception("data not found")
        return self.data

# cargs = ChunkArgs("../Pretrain_Bert", "pretraining", 5)
# cargs.setProxyConfig(1000000, 4769137)
# chunkSave = ChunkSave(cargs)
# chunkSave.load()
# gc.disable()
# # del chunkSave
# # gc.disable()
# print("finished")
# train_example_total = chunkSave.getdata()
# print("get data finished")

# chunkSave.load_part(0)
# train_example_0 = chunkSave.getdata()
# print(getrefcount(train_example_0))
# print(len(train_example_0))
# chunkSave.load_part(1)
# train_example_1 = chunkSave.getdata()
# print(len(train_example_1))
# chunkSave.load_part(2)
# train_example_2 = chunkSave.getdata()
# print(len(train_example_2))
#
# chunkSave.load_part(3)
# train_example_3 = chunkSave.getdata()
# print(len(train_example_3))
# # train_example_0 = [[1,2,3],[1,5,6]]
# # train_example_1 = [[2,2,3],[2,5,6]]
# # train_example_2 = [[3,2,3],[3,5,6]]
# start_time = time.time()
# train_example = train_example_0 * 3
# end_time = time.time()
# print("malloc cost:", end_time - start_time)
# train_example[:len(train_example_0)] = train_example_0
# end_time = time.time()
# print("train 0 cost:", end_time - start_time)
# train_example[:len(train_example_1)] = train_example_1
# end_time = time.time()
# print("train 1 cost:", end_time - start_time)
# train_example[:len(train_example_2)] = train_example_2
# end_time = time.time()
# print("train 2 cost:", end_time - start_time)
# print(len(train_example))
#
# del train_example
# start_time = time.time()
#
# train_example =[]
# train_example.extend(train_example_0)
# train_example.extend(train_example_1)
# train_example.extend(train_example_2)
# train_example.extend(train_example_3)
# end_time = time.time()
# print("extend cost:", end_time - start_time)
# print(len(train_example))
#
# del train_example
# train_example =[]
# for i in range(4):
#     train_example_tmp = chunkSave.load_part(i)
#     start_time = time.time()
#     gc.disable()
#     train_example.extend(train_example_tmp)
#     gc.enable()
#     end_time = time.time()
#     print("extend cost:", end_time - start_time)
# print(len(train_example))
# gc.collect()
# train_test = [1,3,4]
# print(gc.get_referrers(train_test))
# gc.disable()
# train_2 = train_test
# del train_2
# # del train_2
# gc.enable()
#
# print(getrefcount(train_test))
