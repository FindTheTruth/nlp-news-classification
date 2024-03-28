import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
# import torch.optim.adamw
import model_args as args


class MyOptimizer:
    def __init__(self, model_parameters, steps):
        # self.all_params = []
        self.optims = []
        self.schedulers = []

        for name, parameters in model_parameters.items():
            if name.startswith("bert"):
                optim_bert = AdamW(parameters, args.bert_training_lr, eps=1e-8)
                self.optims.append(optim_bert)

                scheduler_bert = get_linear_schedule_with_warmup(optim_bert, 0, steps)
                self.schedulers.append(scheduler_bert)

            elif name.startswith("basic"):
            # if name.startswith("basic"):
                optim = torch.optim.Adam(parameters, lr=args.training_learning_rate,weight_decay=1e-2)
                self.optims.append(optim)

                # l = lambda step: args.decay ** (step // args.decay_step)
                # l = lambda step: 0 if step < args.decay_step else 1
                # scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=l)
                # # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim,)

                scheduler = get_linear_schedule_with_warmup(optim, 0.1, steps)
                # CosineAnnealingWarmRestarts(optim)
                self.schedulers.append(scheduler)
                # self.all_params.extend(parameters)
            else:
                raise Exception("not found paramter dict", name)

        self.num = len(self.optims)

    def step(self):
        for optim, scheduler in zip(self.optims, self.schedulers):
            optim.step()
            scheduler.step()
            optim.zero_grad()

    def zero_grad(self):
        for optim in self.optims:
            optim.zero_grad()

    def get_lr(self):
        lrs = tuple(map(lambda x: x.get_lr()[-1], self.schedulers))
        lr = ' %.5f' * self.num
        res = lr % lrs
        return res
