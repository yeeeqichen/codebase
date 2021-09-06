from pytorch_transformers import BertModel
import torch
import torchsnooper


class ClassifyModel(torch.nn.Module):
    def __init__(self, bert_path, hidden_size, num_class):
        super(ClassifyModel, self).__init__()
        self.bert_encoder = BertModel.from_pretrained(bert_path)
        for param in self.bert_encoder.parameters():
            param.requires_grad = True
        self.non_linear_fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, num_class),
            torch.nn.PReLU()
        )

    def forward(self, inputs):
        tokens = inputs[0]
        mask = inputs[2]
        _, pooled = self.bert_encoder(tokens, attention_mask=mask)
        output = self.non_linear_fc(pooled)
        return output


