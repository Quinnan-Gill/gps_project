from datetime import datetime
from typing import List

from sqlalchemy.sql.expression import label
import torch
from torch.tensor import Tensor

# Dateime Parsing String
DATETIME_PSTR = '%Y_%m_%d'

def _decode_time_str(time):
    return datetime.strptime(time, DATETIME_PSTR)

class PredIndexAccuracy(object):
    def __init__(self, device):
        self.pred_correct = torch.tensor(0).to(device)
        self.pred_incorrect_zeros = torch.tensor(0).to(device)
        self.pred_incorrect_ones = torch.tensor(0).to(device)

    def update(self, preds, labels):
        if len(labels.shape) != 1:
            labels = labels.squeeze(1)
        
        assert len(labels.shape) == 1
        
        bincount = torch.bincount((preds - labels) + 1)

        if len(bincount) == 0 or len(bincount) > 3:
            raise ValueError("Preds returned greater than 2")


        if len(bincount) >= 1:
            # prediction was 0 but label was 1
            # therefore (0 - 1) + 1 == 0
            self.pred_incorrect_zeros += bincount[0]
        if len(bincount) >= 2:
            # prediction was 0/1 and label was 0/1 respectively
            # therefore (0/1 - 0/1) + 1 == 1
            self.pred_correct += bincount[1]
        if len(bincount) >= 3:
            # prediction was 1 but label was 0
            # therefore (1 - 0) + 1 == 2
            self.pred_incorrect_ones += bincount[2]

def safe_bincount(bincount, device):
    if len(bincount) == 0 or len(bincount) > 3:
        raise ValueError("Preds returned greater than 2")

    results = [torch.tensor(0)]*3

    if len(bincount) >= 1:
        # prediction was 0 but label was 1
        # therefore (0 - 1) + 1 == 0
        results[0] = bincount[0].item()
    if len(bincount) >= 2:
        # prediction was 0/1 and label was 0/1 respectively
        # therefore (0/1 - 0/1) + 1 == 1
        results[1] = bincount[1].item()
    if len(bincount) >= 3:
        # prediction was 1 but label was 0
        # therefore (1 - 0) + 1 == 2
        results[2] = bincount[2].item()

    return results


class RunResults(object):
    def __init__(self, device):
        # self.loss: Tensor = torch.tensor(0.0).to(device)
        # if phase == 'train':
        #     self.loss.requires_grad = True
        self.corrects: Tensor = torch.tensor(0).to(device)
        self.loss_list: List[Tensor] = []
