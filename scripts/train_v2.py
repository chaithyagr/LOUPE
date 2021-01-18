"""
    LOUPE training example (v2)

    By Cagla Deniz Bahadir, Adrian V. Dalca and Mert R. Sabuncu
    Primary mail: cagladeniz94@gmail.com
    
    Please cite the below paper for the code:
    
    Bahadir, Cagla Deniz, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Learning-based Optimization of the Under-sampling Pattern in MRI." 
    IPMI 2019
    arXiv preprint arXiv:1901.01960 (2019).
"""

# imports
import os

import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import keras.models

# loupe
from loupe import models


###############################################################################
# parameters
###############################################################################

# TODO put them in the form of ArgumentParser()
#   see e.g. https://github.com/voxelmorph/voxelmorph/blob/master/src/train.py
gpu_id = 0  # gpu id
models_dir = '../models/v2_test/' # change this to a location to save models
nb_epochs_train = 1
batch_size = 32
sparsity = 0.1


###############################################################################
# GPU
###############################################################################

# gpu handling
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)


###############################################################################
# Data - FASHION_MNIST for demo, replace with your favorite dataset
###############################################################################

from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
xdata = np.pad(x_train, ((0,0), (2,2), (2,2)), 'constant')  # get to 32x32
xdata = xdata[..., np.newaxis]/255
val_data = xdata[0:1,...]
xdata = xdata[1:,...]
vol_size = xdata.shape[1:-1]


###############################################################################
# Prepare model
###############################################################################

# load model
model = models.loupe_model(input_shape=vol_size + (1,), filt=64, kern=3,
                           model_type='v2', sparsity=sparsity)

# compile
model.compile(optimizer='Adam', loss='mae')

# prepare save folder
if not os.path.isdir(models_dir): os.makedirs(models_dir)
filename = os.path.join(models_dir, 'model.{epoch:02d}.hdf5')


###############################################################################
# Train model
###############################################################################

# training
#model.save_weights(filename.format(epoch=0))
history = model.fit(xdata, xdata,
                    validation_split=0.3,
                    initial_epoch=1,
                    epochs=1 + nb_epochs_train,
                    batch_size=batch_size, 
                    verbose=1)


###############################################################################
# View prob mask
###############################################################################

mask_filename = os.path.join(models_dir, 'mask.npy')
print('saving mask to %s' % mask_filename)
mask_model = keras.models.Model(model.inputs, model.get_layer('prob_mask_scaled').output)
prob_mask = mask_model.predict(val_data)
np.save(mask_filename, prob_mask)
