from tqdm import tqdm
from absl import app, flags
from torch.utils.data import DataLoader

from utils import _decode_time_str
from datasets import (
    BubbleDataset,
    BubbleDatasetFTP,
    TEC_MEASUREMENTS,
    IBI_MEASUREMENT,
    expand_measurements,
)

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
flags.DEFINE_integer('prefetch', 5000, 'How much to cache for the data')
flags.DEFINE_string('message', '', 'Message for wandb')
flags.DEFINE_enum('label', 'index', IBI_MEASUREMENT.keys(),
                    'Specifies the label for calculating the loss')
flags.DEFINE_enum('rnn_module', 'lstm',
                                    ['lstm', 'peep', 'coupled'],
                                    'Specifies the recurrent module in the RNN.')


def stream_dataset():
    train_dataset = BubbleDataset(
        start_time=_decode_time_str(FLAGS.start_train_time),
        end_time=_decode_time_str(FLAGS.end_train_time),
        bubble_measurements=IBI_MEASUREMENT[FLAGS.label],
        window_size=FLAGS.window_size,
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
    data_loader = train_loader

    progress_bar = tqdm(enumerate(data_loader))
    for step, (sequences, labels) in progress_bar:
        import pdb
        pdb.set_trace()


def main(unused_argvs):
    stream_dataset()

if __name__ == '__main__':
    app.run(main)