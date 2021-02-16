import tensorflow as tf
import tifffile
import numpy as np
from pycromanager import Bridge
from matplotlib import pyplot as plt


def deep_remove():
    model_path = r'C:\Users\PC1\Desktop\python_GUI\model\unet_background.hdf5'
    model = tf.keras.models.load_model(model_path)
    
    bridge = Bridge()
    core = bridge.get_core()
    try:
        while True:
            core.snap_image()
            tagged_image = core.get_tagged_image()
            image = np.reshape(tagged_image.pix, newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])
            bg = model.predict((image[np.newaxis, :, :, np.newaxis])/65535)
            result_img = (image - (bg[0, :, :, 0]*65535))
            plt.imshow(result_img, cmap='gray', vmin=int(np.amin(result_img)), vmax=int(np.amax(result_img)))
            ax = plt.gca()
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            plt.tight_layout()
            plt.pause(0.1)
            plt.cla()
    except:
        pass


if __name__ == '__main__':
    deep_remove()
