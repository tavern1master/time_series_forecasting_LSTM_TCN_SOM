import math
import numpy as np
import torch
from torch.nn import Linear, Module, Sequential, LSTM, Dropout, ReLU


class AutoregressiveLSTM(Module):
    def __init__(self, input_size, hidden_size, n_layers: int=1, dropout: float=0.5):
        super(AutoregressiveLSTM, self).__init__()
        self.hidden_size, self.n_layers = hidden_size, n_layers
        self.lstm = LSTM(input_size=input_size, hidden_size=self.hidden_size,
                         num_layers=self.n_layers, batch_first=True,
                         dropout=dropout if self.n_layers > 1 else 0.0)
        self.dense = Sequential(ReLU(), Dropout(dropout), Linear(self.hidden_size, input_size))

    def init_hidden(self, batch_size: int=0):
        weight = next(self.parameters()).data
        if batch_size:
            hidden = weight.new(self.n_layers, batch_size, self.hidden_size)
            cell = weight.new(self.n_layers, batch_size, self.hidden_size)
        else:
            hidden = weight.new(self.n_layers, self.hidden_size)
            cell = weight.new(self.n_layers, self.hidden_size)
        return hidden.zero_(), cell.zero_()
    
    def forward(self, input: torch.Tensor, hidden):
        out, hidden = self.lstm(input, hidden)
        return self.dense(out[..., [-1], :]), hidden

    def forecast(self, input: torch.Tensor, hidden, seq_length: int=1):
        x, hidden = self.forward(input, hidden)
        preds = [x,]
        for _ in range(1, seq_length):
            out, hidden = self.lstm(x, hidden)
            x = self.dense(out)
            preds.append(x)
        return torch.cat(preds, dim=-2), hidden
    
    def forecast_evaluation(self, input: torch.Tensor, hidden, target_size: int=1):
        x, hidden = self.forward(input, hidden)
        if x.shape[1] >= target_size:
            preds = x.squeeze()
            if preds.ndim > 0:
                return preds[:target_size]
            return preds
        preds = [x,]
        horizon = math.ceil(target_size / x.shape[1])
        temp = np.append(input.cpu().detach().numpy(), x.cpu().detach().numpy())
        for _ in range(1, horizon):
            input = np.delete(temp, np.arange(0,x.shape[1]))
            input = torch.tensor(input, dtype=torch.float32)
            input = input.view(input.shape + (1,)).unsqueeze(1)
            input = input.view(input.shape[1], input.shape[0], input.shape[2])
            out, hidden = self.lstm(input, hidden)
            x = self.dense(out)
            preds.append(x)
            temp = np.append(input.cpu().detach().numpy(), x.cpu().detach().numpy())
        preds = torch.cat(preds, dim=-2)
        return preds.squeeze()[:target_size]