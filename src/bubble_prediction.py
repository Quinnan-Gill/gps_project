import collections
import copy
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import app, flags
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
from unittest.mock import MagicMock

from utils import (
    DATETIME_PSTR,
    PredIndexAccuracy,
    _decode_time_str
)
from datasets import (
    BubbleDataset,
    BubbleDatasetFTP,
    TEC_MEASUREMENTS,
    IBI_MEASUREMENT,
    expand_measurements,
)
from rnn_modules import (
    LSTMCell,
    PeepholedLSTMCell,
    CoupledLSTMCell,
)

import wandb

FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flags.DEFINE_float('weight_decay', 0, 'Weight decay (L2 regularization).')
flags.DEFINE_integer('batch_size', 2048, 'Number of examples per batch.')
flags.DEFINE_integer('epochs', 20, 'Number of epochs for training.')
flags.DEFINE_integer('window_size', 120, 'How large the time window will be')
flags.DEFINE_integer('step_size', 120, 'How much the window shifts')
flags.DEFINE_string('experiment_name', 'exp', 'Defines experiment name.')
flags.DEFINE_string('model_checkpoint', '',
                                        'Specifies the checkpont for analyzing.')
flags.DEFINE_boolean('link', False, 'Links wandb account')
flags.DEFINE_boolean('check_results', False, 'Does a (slow) sanity check on the data correctness')
flags.DEFINE_string('start_train_time', '2016_01_01', 'The start datetime for training')
flags.DEFINE_string('end_train_time', '2016_01_02', 'The end datetime for training')
flags.DEFINE_string('start_val_time', '2017_01_01', 'The start datetime for evaluation')
flags.DEFINE_string('end_val_time', '2017_01_02', 'The end datetime for evaluation')
flags.DEFINE_integer('hidden_size', 50, 'Dimensionality for recurrent neuron.')
flags.DEFINE_string('message', '', 'Message for wandb')
flags.DEFINE_enum('label', 'index', IBI_MEASUREMENT.keys(),
                    'Specifies the label for calculating the loss')
flags.DEFINE_enum('rnn_module', 'lstm',
                                    ['lstm', 'peep', 'coupled'],
                                    'Specifies the recurrent module in the RNN.')

# RNN Modules for LSTM
RNN_MODULES = {
    'lstm': LSTMCell,
    'peep': PeepholedLSTMCell,
    'coupled': CoupledLSTMCell,
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

def bubble_trainer():
    if FLAGS.link:
        WANDB = wandb
    else:
        WANDB = MagicMock()

    train_dataset = BubbleDatasetFTP(
        start_time=_decode_time_str(FLAGS.start_train_time),
        end_time=_decode_time_str(FLAGS.end_train_time),
        bubble_measurements=IBI_MEASUREMENT[FLAGS.label],
        window_size=FLAGS.window_size,
        step_size=FLAGS.step_size,
        index_filter=None,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        num_workers=1
    )
    
    val_dataset = BubbleDatasetFTP(
        start_time=_decode_time_str(FLAGS.start_val_time),
        end_time=_decode_time_str(FLAGS.end_val_time),
        bubble_measurements=IBI_MEASUREMENT[FLAGS.label],
        window_size=FLAGS.window_size,
        step_size=FLAGS.step_size,
        index_filter=None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        num_workers=1
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
        name="{}_{}_{}_{}_{}".format(
            FLAGS.label,
            FLAGS.rnn_module,
            FLAGS.window_size,
            FLAGS.step_size,
            FLAGS.message,
        ),
        reinit=True
    )
    
    model = BubblePredictor(
        rnn_module=RNN_MODULES[FLAGS.rnn_module],
        hidden_size=FLAGS.hidden_size,
    )

    model.to(device)
    WANDB.watch(model, log=None)
    
    print("Device: {}".format(device))
    print('Model Architecture:\n%s' % model)

    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=FLAGS.learning_rate,
        weight_decay=FLAGS.weight_decay
    )
    best_acc = 0.0

    try:
        for phase in ('train', 'eval'):
            if phase == 'train':
                model.train()
                dataset = train_dataset
                data_loader = train_loader
            else:
                model.eval()
                dataset = val_dataset
                data_loader = val_loader
            print("Length of dataset for {}: {} {} {}".format(
                phase, dataset.size, dataset.start_time, dataset.end_time
            ))
            for epoch in range(FLAGS.epochs):

                num_steps = len(data_loader)
                running_loss = 0.0
                running_corrects = 0

                predindex = PredIndexAccuracy(device)

                progress_bar = tqdm(enumerate(data_loader))
                for step, (sequences, labels) in progress_bar:
                    total_step = epoch * len(data_loader) + step

                    sequences = sequences.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs, _ = model(sequences)

                        _, o_w, _ = outputs.shape
                        _, l_w, _ = labels.shape
                        assert o_w == l_w

                        loss = torch.tensor(0.0).to(device)
                        corrects = torch.tensor(0).to(device)
                        loss_list = []
                        for history in range(o_w):
                            output = outputs[:, history, :]
                            label = labels[:, history, :]

                            _, preds = torch.max(output, 1)
                            curr_loss = criterion(output, label.squeeze(1))
                            if device == 'cuda:0':
                                curr_loss = curr_loss.item()
                            loss += curr_loss
                            loss_list.append(curr_loss)
                            corrects += torch.sum(preds == label.data)
                            predindex.update(preds, label.data)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                            # writer.add_scalar('loss', loss.item(), total_step)
                            # writer.add_scalar('accuracy', corrects.item() / len(labels), total_step)
                            progress_bar.set_description(
                                'Step: %d/%d, Loss: %.4f, Loss per: %.4f, Accuracy: %.4f, Epoch %d/%d' %
                                (step, num_steps, loss.item(), loss.item() / o_w, corrects.item() / len(labels), epoch, FLAGS.epochs)
                            )
                            if step % 10 == 0:
                                WANDB.log({
                                    'Training Total Loss': loss.item(),
                                    'Training Accuracy': corrects.item() / (len(labels) * o_w),
                                    'Training Loss Per Chunk': loss.item() / o_w,
                                    'Training Incorrect Zero Guesses': predindex.pred_incorrect_zeros,
                                    'Training Correct Guesses': predindex.pred_correct,
                                    'Training Incorrect One Guesses': predindex.pred_incorrect_ones,
                                })
                            if torch.isnan(loss):
                                    # print("Gradient Explosion, restart run")
                                    sys.exit(1)
                        else:
                            progress_bar.set_description(
                                'Step: %d/%d, Loss: %.4f, Loss per: %.4f, Accuracy: %.4f, Epoch %d/%d' %
                                (step, num_steps, loss.item(), loss.item() / o_w, corrects.item() / len(labels), epoch, FLAGS.epochs)
                            )
                            if step % 10 == 0:
                                WANDB.log({
                                    'Eval Total Loss': loss.item(),
                                    'Eval Accuracy': corrects.item() / (len(labels) * o_w),
                                    'Eval Loss Per Chunk': loss.item() / o_w,
                                    'Eval Incorrect Zero Guesses': predindex.pred_incorrect_zeros,
                                    'Eval Correct Guesses': predindex.pred_correct,
                                    'Eval Incorrect One Guesses': predindex.pred_incorrect_ones,
                                })
                
                    running_loss += loss.item() * sequences.size(0)
                    running_corrects += corrects

                epoch_loss = running_loss / len(dataset)
                epoch_acc = running_corrects / len(dataset)
                WANDB.log({"Epoch Training Accurancy": epoch_acc, "Epoch Training Loss": epoch_loss})
                # print('[Epoch %d] %s accuracy: %.4f, loss: %.4f' %
                #             (epoch + 1, phase, epoch_acc, epoch_loss))
                
                if phase == 'eval':
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model = copy.deepcopy(model.state_dict())
                        # model_copy = copy.deepcopy(model.state_dict())
                        torch.save({
                            'model': best_model,
                        }, os.path.join(experiment_name, 'model_epoch_%d.pt' % (epoch + 1)))

    except KeyboardInterrupt:
        pass

    # final_model = copy.deepcopy(model_copy)
    # torch.save({
    #     'model': final_model
    # }, os.path.join(experiment_name, 'best_model.pt'))

    return

def main(unused_argvs):
    bubble_trainer()

if __name__ == '__main__':
    app.run(main)
