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
from tqdm import tqdm
from datetime import datetime
from unittest.mock import MagicMock

from utils import (
    DATETIME_PSTR,
    PredIndexAccuracy,
    _decode_time_str,
    safe_bincount,
)
from datasets import (
    BubbleDataset,
    BubbleDatasetFTP,
    TEC_MEASUREMENTS,
    IBI_MEASUREMENT,
    expand_measurements,
)
# from models import KeywordSearch
from keywordsearch import UNet, dice_loss

import wandb

FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flags.DEFINE_float('weight_decay', 0, 'Weight decay (L2 regularization).')
flags.DEFINE_integer('batch_size', 2048, 'Number of examples per batch.')
flags.DEFINE_integer('epochs', 20, 'Number of epochs for training.')
flags.DEFINE_integer('window_size', 16, 'The size of the height and width for the image')
flags.DEFINE_integer('step_size', 64, 'How much the window shifts')
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
flags.DEFINE_integer('prefetch', 5000, 'How much to cache for the data')
flags.DEFINE_string('message', '', 'Message for wandb')
flags.DEFINE_enum('label', 'index', IBI_MEASUREMENT.keys(),
                    'Specifies the label for calculating the loss')

def sentence_to_image(sentence):
    batch_size, sentence_len, channels = sentence.size()
    
    image_len = int(sentence_len ** 0.5)

    image = sentence.transpose(1, 2)
    image = image.view(batch_size, channels, image_len, image_len)

    return image

def bubble_image():
    if FLAGS.link:
        WANDB = wandb
    else:
        WANDB = MagicMock()

    train_dataset = BubbleDataset(
        start_time=_decode_time_str(FLAGS.start_train_time),
        end_time=_decode_time_str(FLAGS.end_train_time),
        bubble_measurements=IBI_MEASUREMENT[FLAGS.label],
        window_size=FLAGS.window_size ** 2,
        step_size=FLAGS.step_size,
        index_filter=None,
        prefetch=FLAGS.prefetch,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        num_workers=1
    )
    
    val_dataset = BubbleDataset(
        start_time=_decode_time_str(FLAGS.start_val_time),
        end_time=_decode_time_str(FLAGS.end_val_time),
        bubble_measurements=IBI_MEASUREMENT[FLAGS.label],
        window_size=FLAGS.window_size ** 2,
        step_size=FLAGS.step_size,
        index_filter=None,
        prefetch=FLAGS.prefetch
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
        "CNN",
        FLAGS.hidden_size
    )

    os.makedirs(experiment_name, exist_ok=True)
    writer = SummaryWriter(log_dir=experiment_name)

    WANDB.init(
        project="research",
        name="{}_{}_{}_{}_{}".format(
            FLAGS.label,
            "CNN",
            FLAGS.window_size,
            FLAGS.step_size,
            FLAGS.message,
        ),
        reinit=True
    )
    
    # model = KeywordSearch(
    #     len(expand_measurements(TEC_MEASUREMENTS)),
    #     FLAGS.window_size
    # )
    model = UNet(
        len(expand_measurements(TEC_MEASUREMENTS)),
        2
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
                running_count = 0

                predindex = PredIndexAccuracy(device)

                progress_bar = tqdm(enumerate(data_loader))
                for step, (sequences, labels) in progress_bar:
                    total_step = epoch * len(data_loader) + step

                    # why weird batch sizes
                    sequences = sentence_to_image(sequences).to(device)
                    labels = sentence_to_image(labels).to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(sequences)

                        loss = criterion(outputs, labels.squeeze(1))
                        predindex = safe_bincount(torch.bincount(
                            torch.flatten(
                                torch.subtract(outputs.max(1)[1], labels.squeeze(1))
                            ) + 1
                        ), device)
                        incorrect_ones = predindex[2]
                        corrects = predindex[1]
                        incorrect_zeros = predindex[0]

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                            # writer.add_scalar('loss', loss.item(), total_step)
                            # writer.add_scalar('accuracy', corrects.item() / len(labels), total_step)
                            progress_bar.set_description(
                                'Step: %d/%d, Loss: %.4f, Accuracy: %.4f, Epoch %d/%d' %
                                (step, num_steps, loss.item(), corrects.item() / len(labels), epoch, FLAGS.epochs)
                            )
                            if step % 10 == 0:
                                WANDB.log({
                                    'Training Total Loss': loss.item(),
                                    'Training Total Accuracy': corrects.item(),
                                    'Training Incorrect Zero Guesses': incorrect_zeros.item(),
                                    # 'Training Correct Guesses': predindex.pred_correct / running_count,
                                    'Training Incorrect One Guesses': incorrect_ones.item(),
                                })
                            if torch.isnan(loss):
                                # print("Gradient Explosion, restart run")
                                sys.exit(1)
                        else:
                            progress_bar.set_description(
                                'Step: %d/%d, Loss: %.4f, Accuracy: %.4f, Epoch %d/%d' %
                                (step, num_steps, loss.item(), corrects.item(), epoch, FLAGS.epochs)
                            )
                            if step % 10 == 0:
                                WANDB.log({
                                    'Eval Total Loss': loss.item(),
                                    'Eval Total Accuracy': corrects.item(),
                                    'Eval Incorrect Zero Guesses': incorrect_zeros.item(),
                                    # 'Eval Correct Guesses': predindex.pred_correct,
                                    'Eval Incorrect One Guesses': incorrect_ones.item(),
                                })
                
                    running_loss += loss.item() * sequences.size(0)
                    running_corrects += corrects

                epoch_loss = running_loss
                epoch_acc = running_corrects
                if phase == 'train':
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

    return

def main(unused_argvs):
    bubble_image()

if __name__ == '__main__':
    app.run(main)
