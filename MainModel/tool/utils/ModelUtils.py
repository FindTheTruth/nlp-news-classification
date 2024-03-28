import torch
import os
import logging

os.path.join("./")
# from ModelArgs import ModelArgs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
"""
  torch模型的存储和加载
  使用说明：
  初始化利用ModelArgs,这块说明参照ModelArgs.py
  createDir：
        创建模型存储的目录。
        模型的存储格式：模型名_optimizer[优化器名称]_
                        epoch[迭代轮数]_step[当前轮迭代次数]
                        _lr[初始化学习率]_batch[batch数目]_criterion[评判指标]  
    
"""


class ModelUtils:
    def __init__(self, args):
        self.args = args

    def createDir(self):
        dirname = self.args.modelName + \
                  "_" + "optimizer[" + self.args.optimizer + \
                  "]_" + "epoch" + str(self.args.epoch) + \
                  "_" + "step" + str(self.args.step) + \
                  "_" + "lr" + str(self.args.lr) + \
                  "_" + "batch" + str(self.args.batch) + \
                  "_" + "criterion" + str(self.args.criterion)
        dirPath = os.path.join(self.args.path + "/", dirname)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        return dirPath + "/"

    def getDir(self):
        dirname = self.args.modelName + \
                  "_" + "optimizer[" + self.args.optimizer + \
                  "]_" + "epoch" + str(self.args.epoch) + \
                  "_" + "step" + str(self.args.step) + \
                  "_" + "lr" + str(self.args.lr) + \
                  "_" + "batch" + str(self.args.batch) + \
                  "_" + "criterion" + str(self.args.criterion)
        dirPath = os.path.join(self.args.path + "/", dirname)
        return dirPath + "/"

    def saveModel(self, model):
        dir_path = self.createDir()
        logger.info("saving model")
        torch.save(model.state_dict(), os.path.join(dir_path, "model.pth"))
        logger.info("finish saving model")

    def saveOptimizer(self, optimizer):
        dir_path = self.createDir()
        logger.info("saving optimizer")
        torch.save(optimizer.state_dict(), os.path.join(dir_path, "optimizer.pth"))
        logger.info("finish optimizer saving")

    def saveScheduler(self, scheduler):
        dir_path = self.createDir()
        logger.info("saving scheduler")
        torch.save(scheduler.state_dict(), os.path.join(dir_path, "scheduler.pth"))
        logger.info("finish scheduler saving")

    def loadModel(self, model, pointedPath=None):
        if pointedPath is None:
            dir_path = self.getDir()
        else:
            dir_path = pointedPath + "/"
        logger.info("loading model")
        model_weights = torch.load(os.path.join(dir_path, "model.pth"))
        logger.info(model_weights)
        # print(model_weights)
        model.load_state_dict(model_weights)
        logger.info("finish loading model")
        model.eval()
        return model

    def loadOptimizer(self, optimizer, pointedPath=None):
        if pointedPath is None:
            dir_path = self.getDir()
        else:
            dir_path = pointedPath + "/"
        logger.info("loading optimizer")
        optimizer_weights = torch.load(os.path.join(dir_path, "optimizer.pth"))
        logger.info(optimizer_weights)
        optimizer.load_state_dict(optimizer_weights)
        logger.info("finish loading optimizer")
        return optimizer

    def loadScheduler(self, scheduler, pointedPath=None):
        if pointedPath is None:
            dir_path = self.getDir()
        else:
            dir_path = pointedPath + "/"
        logger.info("loading scheduler")
        scheduler_setting = torch.load(os.path.join(dir_path, "scheduler.pth"))
        logger.info(scheduler_setting)
        scheduler.load_state_dict(scheduler_setting)
        logger.info("finish loading scheduler")
        return scheduler

    def loadBestModel(self, model, optimizer, scheduler):
        current_dir = self.args.path
        best_score = -1
        best_path = ""
        for folder in os.listdir(current_dir):
            if os.path.isdir(folder):
                if folder.find("criterion") != -1:
                    score = float(folder[folder.find("criterion") + 9:])
                    if score > best_score:
                        best_path = os.path.join(self.args.path + "/", folder)
                        best_score = score
        logger.info("loading best model path %s", best_path)
        return self.loadModel(model, best_path), self.loadOptimizer(optimizer, best_path), self.loadScheduler(scheduler,
                                                                                                              best_path)
