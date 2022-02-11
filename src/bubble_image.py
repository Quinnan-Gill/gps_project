import random
import copy
import os
import sys

from sqlalchemy.sql.expression import label

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import app, flags
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
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
    TEC_MEASUREMENTS,
    IBI_MEASUREMENT,
    expand_measurements,
)

from expanded_datasets import (
    BubbleDatasetExpanded
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
flags.DEFINE_enum('dataset', 'bubble_dataset', ['bubble_dataset', 'expanded_dataset'], 'What dataset is being used')
flags.DEFINE_boolean('img_capture', False, 'Capture heatmap of the values')
flags.DEFINE_float('img_threshold', 0.2, 'Percentage of the image needing to be ones')

IMAGE_DIR = "./images/"
LOOP_BREAK = 1000

DATASET_DICT = {
    'bubble_dataset': BubbleDataset,
    'expanded_dataset': BubbleDatasetExpanded
}

def one_count(labels):
    num_ones = 0

    label_bincount = torch.bincount(
        torch.flatten(labels.squeeze(1))
    )

    if len(label_bincount) == 2:
        num_ones = label_bincount[1].item()

    return num_ones

def label_size(labels):
    return torch.flatten(labels).shape[0]

class ImageCapture:
    def __init__(self):
        self.captured = False
        self.flag_string = f"{FLAGS.end_train_time}_{FLAGS.window_size}_{FLAGS.step_size}"
        self.softmax = nn.Soft

        self.image_directory = os.path.join(
            IMAGE_DIR, f"cnn_{self.flag_string}"
        )
        
        # Create a directory if it does not exist
        if not os.path.exists(self.image_directory):
            os.makedirs(self.image_directory)
            
    def select_image(self, label_batch):
        batch_size = label_batch.shape[0]
        size = label_size(label_batch[0])
        i = 0

        while True:
            random_selection = random.randint(0, batch_size)
            
            labels = label_batch[random_selection]
            
            num_ones = one_count(labels)
            
            if num_ones / size > FLAGS.img_threshold:
                return random_selection

            i += 1
            if i > LOOP_BREAK:
                print("Could not find good image")
                return random_selection
            
    
    def save_image(self, image_data, labels):
        self.captured = True

        assert len(image_data.shape) == 3, "Only send one image in batch"
        
        image_labels = expand_measurements(TEC_MEASUREMENTS)
        for i, channel in enumerate(image_data):
            ax = sns.heatmap(channel, linewidth=0.5)
            image_name = os.path.join(self.image_directory, f"{image_labels[i]}_{self.flag_string}")
            plt.savefig(image_name)

        ax = sns.heatmap(labels.squeeze(0), linewidth=0.5)
        image_name = os.path.join(self.image_directory, f"labels_{self.flag_string}")
        plt.savefig(image_name)

    def select_and_save_image(self, image_batch, label_batch):
        if self.captured:
            return

        rand_sel = self.select_image(label_batch)
        
        image_data = image_batch[rand_sel]
        labels = label_batch[rand_sel]

        self.save_image(image_data, labels)


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

    if FLAGS.img_capture:
        image_capture = ImageCapture()
    else:
        image_capture = MagicMock()

    train_dataset = DATASET_DICT[FLAGS.dataset](
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
    
    val_dataset = DATASET_DICT[FLAGS.dataset](
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
        project="cnn_research",
        name="{}_{}_{}_{}_{}".format(
            FLAGS.label,
            "CNN",
            FLAGS.window_size,
            FLAGS.step_size,
            FLAGS.message,
        ),
        reinit=True
    )

    model = UNet(
        val_dataset.get_column_size(),
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
                running_total = 0.0

                predindex = PredIndexAccuracy(device)

                progress_bar = tqdm(enumerate(data_loader))
                for step, (sequences, labels) in progress_bar:
                    total_step = epoch * len(data_loader) + step

                    # why weird batch sizes
                    sequences = sentence_to_image(sequences).to(device)
                    labels = sentence_to_image(labels).to(device)

                    optimizer.zero_grad()

                    image_capture.select_and_save_image(sequences, labels)

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(sequences)

                        num_ones = one_count(labels)

                        loss = criterion(outputs, labels.squeeze(1))
                        # predindex = safe_bincount(torch.bincount(
                        #     torch.flatten(
                        #         torch.subtract(outputs.max(1)[1], labels.squeeze(1))
                        #     ) + 1
                        # ), device)
                        
                        numpy_outputs = outputs.max(1)[1].cpu().numpy()
                        numpy_labels = labels.squeeze(1).cpu().numpy()

                        corrects = np.sum(numpy_outputs == numpy_labels)
                        incorrects = np.sum(numpy_outputs != numpy_labels)

                        correct_ones = np.sum(
                            np.logical_and(
                                numpy_outputs == numpy_labels,
                                numpy_labels == 1
                            )
                        )
                        incorrect_ones = np.sum(
                            np.logical_and(
                                numpy_outputs != numpy_labels,
                                numpy_labels == 1
                            )
                        )

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                            # writer.add_scalar('loss', loss.item(), total_step)
                            # writer.add_scalar('accuracy', corrects.item() / len(labels), total_step)
                            progress_bar.set_description(
                                'Step: %d/%d, Loss: %.4f, Accuracy: %.4f, Accuracy %%: %.4f%%, Epoch %d/%d' %
                                (step, num_steps, loss.item(), corrects, corrects / float(corrects + incorrects), epoch, FLAGS.epochs)
                            )
                            if step % 10 == 0:
                                WANDB.log({
                                    'Training Loss': loss.item(),
                                    'Training Accuracy': corrects,
                                    'Training Accuracy Precent': corrects / float(corrects + incorrects),
                                    'Training Accuracy for ones': correct_ones,
                                    'Training Accuracy for ones Percent': correct_ones / float(correct_ones + incorrect_ones)
                                    # 'Training Incorrect Zero Guesses': incorrect_zeros.item(),
                                    # # 'Training Correct Guesses': predindex.pred_correct / running_count,
                                    # 'Training Incorrect One Guesses': incorrect_ones.item(),
                                #     'Training Ones': num_ones
                                })
                            if torch.isnan(loss):
                                print("Gradient Explosion, restart run")
                                sys.exit(1)
                        else:
                            progress_bar.set_description(
                                'Step: %d/%d, Loss: %.4f, Accuracy: %.4f, Epoch %d/%d' %
                                (step, num_steps, loss.item(), corrects.item(), epoch, FLAGS.epochs)
                            )
                            if step % 10 == 0:
                                WANDB.log({
                                    'Eval Loss': loss.item(),
                                    'Eval Accuracy': corrects,
                                    'Eval Accuracy Precent': corrects / float(corrects + incorrects),
                                    'Eval Accuracy for ones': correct_ones,
                                    'Eval Accuracy for ones Percent': correct_ones / float(correct_ones + incorrect_ones)
                                    # 'Eval Incorrect Zero Guesses': incorrect_zeros.item(),
                                    # 'Eval Correct Guesses': predindex.pred_correct,
                                    # 'Eval Incorrect One Guesses': incorrect_ones.item(),
                                })
                
                    running_loss += loss.item() * sequences.size(0)
                    running_corrects += corrects
                    running_total += float(corrects + incorrects)

                epoch_loss = running_loss
                epoch_acc = running_corrects
                epoch_acc_percent = running_corrects / running_total
                if phase == 'train':
                    WANDB.log({
                        "Epoch Training Accurancy": epoch_acc,
                        "Epoch Training Loss": epoch_loss,
                        "Epoch Training Accuracy Precent": epoch_acc_percent,
                    })
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
