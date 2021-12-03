import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from parameters import *
from utils import RunResults

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

    def __get_device(self):
        device_obj = next(self.parameters()).device
        return device_obj.type

    def evaluate_output(self,
                        outputs,
                        labels,
                        criterion,
                        predindex,
                        phase,
                        loss):

        device = self.__get_device()

        _, ouput_width, _ = outputs.shape
        _, label_width, _ = labels.shape
        assert ouput_width == label_width

        results = RunResults(device)

        # loss = torch.tensor(0.0).to(device)
        # if phase == 'train':
        #     loss.requires_grad = True
        for history in range(ouput_width):
            output = outputs[:, history, :]
            label = labels[:, history, :]

            _, preds = torch.max(output, 1)
            curr_loss = criterion(output, label.squeeze(1))

            if device == 'cuda':
                curr_loss = curr_loss.item()

            loss = loss + curr_loss
            results.corrects += torch.sum(preds == label.data)
            predindex.update(preds, label.data)

        return results, loss

    def reset_parameters(self):
        with torch.no_grad:
            for param in self.parameters():
                param.reset_parameters()
        return

class KeywordSearch(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.model = nn.Sequential()

        self.model.add_module("StartBatchNorm", nn.BatchNorm2d(input_size))

        prev_layer_size = input_size
        # padding = F.pad()
        for i, num_filters in enumerate(filters):
            # Convolutional layers
            self.model.add_module(f"Conv2d-{i+1}", nn.Conv2d(prev_layer_size, num_filters, kernel_size=(1, 1)))
            # self.model.add_module(f"ZeroPad-{i+1}", nn.ZeroPad2d((0, 0, 2, 1)))
            self.model.add_module(f"BatchNorm-{i+1}", nn.BatchNorm2d(num_filters))
            self.model.add_module(f"ReLu-{i+1}", nn.ReLU())

            prev_layer_size = num_filters

            # Pooling
            self.model.add_module(f"MaxPooling2D-{i+1}", nn.MaxPool2d(kernel_size=POOL_SIZE))
            self.model.add_module(f"Dropout-{i+1}", nn.Dropout(DROPOUT))

        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(1024, DENSE_1)
        self.batch1 = nn.BatchNorm1d(DENSE_1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(DROPOUT)
        self.lin2 = nn.Linear(DENSE_1, DENSE_2)
        self.batch2 = nn.BatchNorm1d(DENSE_2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(DROPOUT)
        
        self.classification = nn.Linear(DENSE_2, NUM_CLASSES)


    def forward(self, history, state=None):
        conv = self.model(history)
        import pdb
        pdb.set_trace()
        x = self.flatten(conv)
        x = self.lin1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.lin2(x)
        x = self.batch2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.classification(x)
    
        return x
        

