import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import (
    TEC_MEASUREMENTS,
    expand_measurements,
)

class BubblePredictor(nn.Module):

    def __init__(
        self,
        rnn_module,
        hidden_size,
        bias=False,
    ):
        super().__init__()
        self.rnn_module = rnn_module
        self.hidden_size = hidden_size
        self.bias = bias

        self.rnn_module = self.rnn_module(
            input_size=len(expand_measurements(TEC_MEASUREMENTS)),
            hidden_size=hidden_size,
            bias=bias
        )
        self.bubble_prediction = nn.Linear(hidden_size, 2)

        return
    
    def forward(self, history, state=None):
        batch_size, history_steps, _ = history.shape

        logits = None
        logit_set = False

        for step in range(history_steps):
            state = self.rnn_module(history[:, step, :], state)
        
            if isinstance(state, tuple):
                outputs, _ = state
            else:
                outputs = state
            
            pred = self.bubble_prediction(outputs).unsqueeze(1)
            
            if not logit_set:
                logits = pred
                logit_set = True
            else:
                logits = torch.cat((logits, pred), 1)
        return logits, state

    def reset_parameters(self):
        with torch.no_grad:
            for param in self.parameters():
                param.reset_parameters()
        return

# class KeywordSearch(nn.Model):


