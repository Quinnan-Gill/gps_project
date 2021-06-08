import collections
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import app, flags
from torch.distributions import categorical
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
from unittest.mock import MagicMock

from datasets import (
    BubbleDataset,
    TEC_MEASUREMENTS,
    IBI_MEASUREMENT,
    expand_measurements,
    DATETIME_PSTR,
)
from rnn_modules import LSTMCell

import wandb

FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flags.DEFINE_float('weight_decay', 0, 'Weight decay (L2 regularization).')
flags.DEFINE_integer('batch_size', 2048, 'Number of examples per batch.')
flags.DEFINE_integer('epochs', 20, 'Number of epochs for training.')
flags.DEFINE_string('experiment_name', 'exp', 'Defines experiment name.')
flags.DEFINE_string('model_checkpoint', '',
                                        'Specifies the checkpont for analyzing.')
flags.DEFINE_boolean('link', False, 'Links wandb account')
flags.DEFINE_string('start_time', '2014_01_01', 'The start datetime')
flags.DEFINE_string('end_time', '2014_01_02', 'The end datetime')
flags.DEFINE_integer('hidden_size', 50, 'Dimensionality for recurrent neuron.')
flags.DEFINE_enum('label', 'index', IBI_MEASUREMENT.keys(),
                    'Specifies the label for calculating the loss')
flags.DEFINE_enum('rnn_module', 'lstm',
                                    ['lstm'],
                                    'Specifies the recurrent module in the RNN.')

# RNN Modules for LSTM
RNN_MODULES = {
    'lstm': LSTMCell
}

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
        self.bubble_prediction = nn.Linear(hidden_size, 3).double()

        return
    
    def forward(self, history, state=None):
        batch_size, history_steps = history.shape

        state = self.rnn_module(history, state)
        
        if isinstance(state, tuple):
            outputs, _ = state
        else:
            outputs = state
        
        logits = self.bubble_prediction(outputs)

        return logits, state

    def reset_parameters(self):
        with torch.no_grad:
            for param in self.parameters():
                param.reset_parameters()
        return

def _decode_time_str(time):
    return datetime.strptime(time, DATETIME_PSTR)

def bubble_trainer():
    if FLAGS.link:
        WANDB = wandb
    else:
        WANDB = MagicMock()

    train_dataset = BubbleDataset(
        start_time=_decode_time_str(FLAGS.start_time),
        end_time=_decode_time_str(FLAGS.end_time),
        bubble_measurements=IBI_MEASUREMENT[FLAGS.label],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=8
    )
    
    best_model = None
    best_loss = 0.0
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    experiment_name = 'experiments/{}_{}_{}.h_{}'.format(
        FLAGS.experiment_name,
        FLAGS.label,
        FLAGS.rnn_module,
        FLAGS.hidden_size
    )

    os.makedirs(experiment_name, exist_ok=True)
    writer = SummaryWriter(log_dir=experiment_name)

    WANDB.init(
        project="research",
        name="{}_{}".format(FLAGS.label, FLAGS.rnn_module),
        reinit=True
    )
    
    model = BubblePredictor(
        rnn_module=RNN_MODULES[FLAGS.rnn_module],
        hidden_size=FLAGS.hidden_size,
    )

    model.to(device)
    WANDB.watch(model)
    
    print('Model Architecture:\n%s' % model)

    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=FLAGS.learning_rate,
        weight_decay=FLAGS.weight_decay
    )

    phase = 'train'

    try:
        for epoch in range(FLAGS.epochs):
            model.train()
            dataset = train_dataset
            data_loader = train_loader

            num_steps = len(data_loader)
            running_loss = 0.0
            running_corrects = 0

            progress_bar = tqdm(enumerate(data_loader))
            for step, (sequences, labels) in progress_bar:
                total_step = epoch * len(data_loader) + step

                sequences = sequences.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs, _ = model(sequences)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels.squeeze_(1))
                corrects = torch.sum(preds == labels.data)

                loss.backward()
                optimizer.step()

                writer.add_scalar('loss', loss.item(), total_step)
                writer.add_scalar('accuracy', corrects.item() / len(labels), total_step)
                progress_bar.set_description(
                    'Step: %d/%d, Loss: %.4f, Accuracy: %.4f, Epoch %d/%d' %
                    (step, num_steps, loss.item(), corrects.item() / len(labels), epoch, FLAGS.epochs)
                )
                if step % 100 == 0:
                    WANDB.log({'Training Loss': loss.item(), 'Training Accuracy': corrects.item() / len(labels)})
            
                running_loss += loss.item() * sequences.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataset)
            epoch_acc = running_corrects.double() / len(dataset)
            WANDB.log({"Training Accurancy": epoch_acc, "Training Loss": epoch_loss})
            print('[Epoch %d] %s accuracy: %.4f, loss: %.4f' %
                        (epoch + 1, phase, epoch_acc, epoch_loss))
                

            model_copy = copy.deepcopy(model.state_dict())
            torch.save({
                'model': model_copy,
            }, os.path.join(experiment_name, 'model_epoch_%d.pt' % (epoch + 1)))

    except KeyboardInterrupt:
        pass

    final_model = copy.deepcopy(model_copy)
    torch.save({
        'model': final_model
    }, os.path.join(experiment_name, 'best_model.pt'))

    return

def main(unused_argvs):
    bubble_trainer()

if __name__ == '__main__':
    app.run(main)