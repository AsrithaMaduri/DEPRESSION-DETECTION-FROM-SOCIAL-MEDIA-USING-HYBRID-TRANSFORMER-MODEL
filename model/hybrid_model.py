
import torch
import torch.nn as nn
from transformers import DistilBertModel

class HybridModel(nn.Module):

    def __init__(self, ling_dim=4):
        super().__init__()

        self.bert = DistilBertModel.from_pretrained(
         "distilbert-base-uncased",
           torch_dtype=torch.float32
       )
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.Linear(512,1)

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(512+ling_dim,2)

    def forward(self,input_ids,attention_mask,ling_feat):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        seq = outputs.last_hidden_state

        lstm_out,_ = self.lstm(seq)

        att = torch.softmax(self.attention(lstm_out),dim=1)

        context = torch.sum(att*lstm_out,dim=1)

        combined = torch.cat((context,ling_feat),dim=1)

        out = self.dropout(combined)

        return self.fc(out)
