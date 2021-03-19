"""
Train the model.
"""
from pathlib import Path
import datetime
import argparse
import yaml
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import os 
from os import path
import keras
import keras.backend as K
import keras.losses

from dataset import LandCoverData as LCD
from dataset import parse_image, load_image_train, load_image_test
from model import UNet
from tensorflow_utils import plot_predictions
from utils import YamlNamespace

if not path.exists('/content/experiments'):
  Save_model='/content/experiments'
  os.mkdir(Save_model)

class PlotCallback(tf.keras.callbacks.Callback):
    """A callback used to display sample predictions during training."""
    from IPython.display import clear_output

    def __init__(self, dataset: tf.data.Dataset=None,
                 sample_batch: tf.Tensor=None,
                 save_folder: Path=None,
                 num: int=1,
                 ipython_mode: bool=False):
        super(PlotCallback, self).__init__()
        self.dataset = dataset
        self.sample_batch = sample_batch
        self.save_folder = save_folder
        self.num = num
        self.ipython_mode = ipython_mode

    def on_epoch_begin(self, epoch, logs=None):
        if self.ipython_mode:
            self.clear_output(wait=True)
        if self.save_folder:
            save_filepaths = [self.save_folder/f'plot_{n}_epoch{epoch}.png' for n in range(1, self.num+1)]
        else:
            save_filepaths = None
        plot_predictions(self.model, self.dataset, self.sample_batch, num=self.num, save_filepaths=save_filepaths)

class WeightedSparseCategoricalCrossEntropy(keras.losses.Loss):
    """
    Args:
      class_weight: class_weight defined using class distribution
      from_logits: Whether to compute loss from logits or the probability.
      reduction: Type of tf.keras.losses.Reduction to apply to loss.
      name: Name of the loss function.
    """
    def __init__(self, from_logits=False,
                 reduction=keras.losses.Reduction.AUTO,
                 name='weighted_sparse_categorical_crossentropy'):
        super().__init__(reduction=reduction, name=name)
        #self.class_weight = class_weight
        self.from_logits = from_logits


    def call(self, y_true, y_pred):
        #return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1)
        #scce = tf.keras.losses.SparseCategoricalCrossentropy()
        #return scce(y_true, y_pred)
        log_ = K.mean(K.sparse_categorical_crossentropy(y_true, y_pred))
        return K.sum (log_ * K. constant(class_weight))

class dice_loss(keras.losses.Loss):
    '''
      Dice coefficient for 10 categories. Ignores pixel of label 0 and 1
      Pass to model as metric during compile statement
    '''
    def __init__(self, from_logits=False,
                 reduction=keras.losses.Reduction.AUTO,
                 name='dice_loss'):
        super().__init__(reduction=reduction, name=name)
        self.from_logits = from_logits


    def call(self, y_true, y_pred):
        smooth=1e-7
        y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=10)[...,2:])
        y_pred_f = K.flatten(y_pred[...,2:])
        intersection = K.sum(y_true_f * y_pred_f, axis=-1)
        den = K.sum(y_true_f + y_pred_f, axis=-1)
        dice_coeff = K.mean((2. * intersection / (den + smooth)))
        return 1 - dice_coeff
      
class jaccard_loss(keras.losses.Loss):
    def __init__(self, from_logits=False,
                 reduction=keras.losses.Reduction.AUTO,
                 name='jaccard_loss'):
        super().__init__(reduction=reduction, name=name)
        self.from_logits = from_logits


    def call(self, y_true, y_pred):
        smooth=100
        y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=10)[...,2:])
        y_pred_f = K.flatten(y_pred[...,2:])
        intersection = K.sum(K.abs(y_true_f * y_pred_f), axis=-1)
        sum_ = K.sum(K.abs(y_true_f) + K.abs(y_pred_f), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return (1 - jac) * smooth

def _parse_args():
    parser = argparse.ArgumentParser('Training script')
    parser.add_argument('--config', '-c', type=str, required=True, help="The YAML config file")
    cli_args = parser.parse_args()
    # parse the config file
    with open(cli_args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = YamlNamespace(config)
    config.xp_rootdir = Path(config.xp_rootdir).expanduser()
    assert config.xp_rootdir.is_dir()
    config.dataset_folder = Path(config.dataset_folder).expanduser()
    assert config.dataset_folder.is_dir()
    if config.val_samples_csv is not None:
        config.val_samples_csv = Path(config.val_samples_csv).expanduser()
        assert config.val_samples_csv.is_file()

    return config

if __name__ == '__main__':

    import multiprocessing

    config = _parse_args()
    print(f'Config:\n{config}')
    # set random seed for reproducibility
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        tf.random.set_seed(config.seed)

    N_CPUS = multiprocessing.cpu_count()

    print('Instanciate train and validation datasets')
    train_files = list(config.dataset_folder.glob('train/images/images/*.tif'))
    #train_files = train_files[:500]
    # shuffle list of training samples files
    train_files = random.sample(train_files, len(train_files))
    devset_size = len(train_files)
    # validation set
    if config.val_samples_csv is not None:
        # read the validation samples
        val_samples_s = pd.read_csv(config.val_samples_csv, squeeze=True)
        val_files = [config.dataset_folder/'train/images/images/{}.tif'.format(i) for i in val_samples_s]
        train_files = [f for f in train_files if f not in set(val_files)]
        valset_size = len(val_files)
        trainset_size = len(train_files)
        assert valset_size + trainset_size == devset_size
    else:
        # generate a hold-out validation set from the training set
        valset_size = int(len(train_files) * 0.1)
        train_files, val_files = train_files[valset_size:], train_files[:valset_size]
        trainset_size = len(train_files) - valset_size

    train_dataset = tf.data.Dataset.from_tensor_slices(list(map(str, train_files)))\
        .map(parse_image, num_parallel_calls=N_CPUS)
    val_dataset = tf.data.Dataset.from_tensor_slices(list(map(str, val_files)))\
        .map(parse_image, num_parallel_calls=N_CPUS)

    train_dataset = train_dataset.map(load_image_train, num_parallel_calls=N_CPUS)\
        .shuffle(buffer_size=1024, seed=config.seed)\
        .repeat()\
        .batch(config.batch_size)\
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = val_dataset.map(load_image_test, num_parallel_calls=N_CPUS)\
        .repeat()\
        .batch(config.batch_size)\
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # where to write files for this experiments
    xp_dir = config.xp_rootdir / datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    (xp_dir/'tensorboard').mkdir(parents=True)
    (xp_dir/'plots').mkdir()
    (xp_dir/'checkpoints').mkdir()
    # save the validation samples to a CSV
    val_samples_s = pd.Series([int(f.stem) for f in val_files], name='sample_id', dtype='uint32')
    val_samples_s.to_csv(xp_dir/'val_samples.csv', index=False)

    # keep a training minibatch for visualization
    for image, mask in train_dataset.take(1):
        sample_batch = (image[:5, ...], mask[:5, ...])
    
    strxp_dir = xp_dir.absolute()
    strxp_dir = strxp_dir.as_posix()
    
    callbacks = [
        PlotCallback(sample_batch=sample_batch, save_folder=xp_dir/'plots', num=5),
        tf.keras.callbacks.TensorBoard(
            log_dir=xp_dir/'tensorboard',
            update_freq='epoch'
        ),
        # tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=strxp_dir + '/' +'checkpoints/epoch{epoch}', save_best_only=False, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=(xp_dir/'fit_logs.csv')
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=20,
            factor=0.5,
            verbose=1,
        )
    ]
    # create the U-Net model to train
    unet_kwargs = dict(
        input_shape=(LCD.IMG_SIZE, LCD.IMG_SIZE, LCD.N_CHANNELS),
        num_classes=LCD.N_CLASSES,
        num_layers=2
    )
    print(f"Creating U-Net with arguments: {unet_kwargs}")
    model = UNet(**unet_kwargs)
    print(model.summary())

    # get optimizer, loss, and compile model for training
    optimizer = tf.keras.optimizers.Adam(lr=config.lr)

    # compute class weights for the loss: inverse-frequency balanced
    # note: we set to 0 the weights for the classes "no_data"(0) and "clouds"(1) to ignore these
    class_weight = (1 / LCD.TRAIN_CLASS_COUNTS)* LCD.TRAIN_CLASS_COUNTS.sum() / (LCD.N_CLASSES)
    class_weight[LCD.IGNORED_CLASSES_IDX] = 0.
    print(f"Will use class weights: {class_weight}") 
    
    #loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    loss = WeightedSparseCategoricalCrossEntropy()
    #loss = dice_loss()
    #loss = jaccard_loss()
    print("Compile model")
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[])
                  # metrics = [tf.keras.metrics.Precision(),
                  #            tf.keras.metrics.Recall(),
                  #            tf.keras.metrics.MeanIoU(num_classes=LCD.N_CLASSES)]) # TODO segmentation metrics

    # Launch training
    model.fit(train_dataset, epochs=config.epochs,
                              callbacks=callbacks,
                              steps_per_epoch=trainset_size // config.batch_size,
                              validation_data=val_dataset,
                              validation_steps=valset_size // config.batch_size,
                              )
    model.save('/content/experiments/saved')
