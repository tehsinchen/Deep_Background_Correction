from utils import *
import tifffile
import numpy as np
from model import unet
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import time


start_time = time.time()

# setup directories
foldername = r'C:\Users\NBP\Desktop\20200916'
result_folder = foldername + "\\" + "Analyzed_results"
filetype = ".tif"
train_path = result_folder + "\\" + 'train'
test_path = result_folder + "\\" + 'test'
result_path =test_path + "\\" + 'prediction'
model_path = result_folder + "\\" + 'unet_background.hdf5'

# start training
data_gen_args = dict(rescale=1.0/65535, horizontal_flip=True, vertical_flip=True)
myGene = trainGenerator(5, train_path, 'image', 'truth', data_gen_args, save_to_dir=None)
model = unet()
model_checkpoint = ModelCheckpoint(model_path, monitor='loss', verbose=1, save_best_only=True)
lr_sched = step_decay_schedule(initial_lr=0.01, decay_factor=0.5, step_size=15)
history = model.fit(myGene, steps_per_epoch=500, epochs=2, callbacks=[model_checkpoint, lr_sched])

# generate the predicted results
test_generator = testGenerator(test_path, 'image', 1)
pred = model.predict(test_generator, steps=len(test_generator), verbose=1)

# save the loss history
plt.semilogy(history.history['loss'])
plt.title('loss history')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig(result_folder + "\\" + "loss_history.png")
plt.close()

elapsed_time = time.time() - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))