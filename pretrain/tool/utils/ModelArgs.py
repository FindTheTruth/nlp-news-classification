# -----------ARGS---------------------

"""
  Model的配置文件
      batch: int 表示当前模型的batch数
      lr:float表示当前模型的初始化学习率
      optimizer: str 表示当前模型使用的优化器，取值egg:"adam","sgd"

      step:int 表示当前epoch训练的step
      steps:int 表示预期epoch训练的steps
      epoch:int 表示当前训练的第几个epoch
      epochs:int 表示训练的总epochs

      path:str 表示模型存储的目录 egg:"./train"
      modelName:str 模型存储的命名

      criterion:float 模型训练在验证集上表现的好坏指标
"""


class ModelArgs:
    def __init__(self):
        self._batch = None
        self._lr = None
        self._optimizer = None

        self._step = None
        self._steps = None
        self._epoch = None
        self._epochs = None

        self._path = None
        self._modelName = None

        self._criterion = None



    @property
    def batch(self):
        if self._batch is None:
            raise Exception("batch not set")
        return self._batch

    @batch.setter
    def batch(self, batch: int):
        assert (type(batch) == int)
        self._batch = batch

    @property
    def lr(self):
        if self._lr is None:
            raise Exception("lr not set")
        return self._lr

    @lr.setter
    def lr(self, lr: float):
        assert (type(lr) == float)
        self._lr = lr

    @property
    def optimizer(self):
        if self._optimizer is None:
            raise Exception("optimizer name not set")
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: str):
        assert (type(optimizer) == str)
        self._optimizer = optimizer

    @property
    def step(self):
        if self._step is None:
            raise Exception("step not set")
        return self._step

    @step.setter
    def step(self, step: int):
        assert (type(step) == int)
        self._step = step

    @property
    def steps(self):
        if self._steps is None:
            raise Exception("steps not set")
        return self._steps

    @steps.setter
    def steps(self, steps: int):
        assert (type(steps) == int)
        self._steps = steps

    @property
    def epoch(self):
        if self._epoch is None:
            raise Exception("epoch not set")
        return self._epoch

    @epoch.setter
    def epoch(self, epoch: int):
        assert (type(epoch) == int)
        self._epoch = epoch

    @property
    def epochs(self):
        if self._epochs is None:
            raise Exception("epochs not set")
        return self._epochs

    @epochs.setter
    def epochs(self, epochs: int):
        assert (type(epochs) == int)
        self._epochs = epochs

    @property
    def path(self):
        if self._path is None:
            raise Exception("path not set")
        return self._path

    @path.setter
    def path(self, path: str):
        assert (type(path) == str)
        self._path = path

    @property
    def modelName(self):
        if self._modelName is None:
            raise Exception("model name not set")
        return self._modelName

    @modelName.setter
    def modelName(self, modelName):
        self._modelName = modelName

    @property
    def criterion(self):
        if self._criterion is None:
            raise Exception("criterion not set")
            return self._criterion
        return self._criterion

    @criterion.setter
    def criterion(self, criterion):
        assert (type(criterion) == float)
        self._criterion = criterion
