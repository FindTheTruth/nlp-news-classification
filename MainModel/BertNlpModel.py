import torch
import torch.nn as nn
import model_args as args


class BertNlpModel(torch.nn.Module):
    def __init__(self, preTrain):
        super(BertNlpModel, self).__init__()
        self.preTrain = preTrain
        hidden_size = 512
        class_num = 14

        self.dense_1 = nn.Linear(hidden_size, hidden_size)
        self.layerNorm = nn.LayerNorm(hidden_size)
        self.gelu = nn.GELU()
        self.dense_2 = nn.Linear(hidden_size, class_num)

        self.dropout_ops = nn.ModuleList(
            [nn.Dropout() for _ in range(args.dropout_num)]
        )

        self.dropout = nn.Dropout(0.1)

        self.all_parameters = {}
        parameters = []
        parameters.extend(list(filter(lambda p: p.requires_grad, self.dense_1.parameters())))
        parameters.extend(list(filter(lambda p: p.requires_grad, self.layerNorm.parameters())))
        parameters.extend(list(filter(lambda p: p.requires_grad, self.dense_2.parameters())))
        if len(parameters) > 0:
            self.all_parameters["basic_parameters"] = parameters

        self.all_parameters["bert_parameters"] = self.preTrain.parameters()

    def train(self):
        super(BertNlpModel, self).train()
        self.preTrain.train()

    def eval(self):
        super(BertNlpModel,self).eval()
        # super(BertNlpModel, self).train(False)
        self.preTrain.eval()

    def forward(self, input_ids, segment_ids, input_mask):
        # print("-----------output-----------------------")
        hidden_states, _ = self.preTrain.bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)
        # print(hidden_states.size())
        hidden_states = self.gelu(self.dense_1(hidden_states))

        hidden_states = self.layerNorm(hidden_states)

        hidden_states = torch.mean(hidden_states, dim=1, keepdim=False)
        # print("drop:" ,hidden_states.size())

        if args.multi_dropout:
            for i, dropout_op in enumerate(self.dropout_ops):
                if i == 0:
                    out = dropout_op(hidden_states)

                else:
                    temp_out = dropout_op(hidden_states)
                    out += temp_out
            hidden_states = out / args.dropout_num
        else:
            hidden_states = self.dropout(hidden_states)

        # print("drop:" ,hidden_states.size())

        # hidden_states = torch.mean(hidden_states, dim=1, keepdim=False)

        output = self.dense_2(hidden_states)

        return output




