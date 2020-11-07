import skimage
import skimage.io
import skimage.transform
from matplotlib.colors import SymLogNorm

import os
import numpy as np
import utils
import matplotlib.pyplot as plt
from task4b import convolve_im


if __name__ == "__main__":
    # DO NOT CHANGE
    impath = os.path.join("images", "noisy_moon.png")
    im = utils.read_im(impath)

    # START YOUR CODE HERE ### (You can change anything inside this block)
    im_freq = np.fft.fft2(im)
    im_freq_old = im_freq.copy()

    neigboor_average = np.ones((5, 5))/25
    neigboor_average[2, 2] = 0
    neigboor_convolved = convolve_im(np.abs(im_freq), neigboor_average, False)
    ratio = np.abs(im_freq) / neigboor_convolved
    noise_arg = np.unravel_index(
        np.argmax(ratio), im_freq.shape)
    print(noise_arg)

    im_freq[noise_arg] = 0
    im_freq[-noise_arg[0], -noise_arg[1]] = 0
    im_filtered = np.fft.ifft2(im_freq)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(im, cmap="gray")
    ax[0, 1].imshow(np.abs(im_filtered), cmap="gray")
    # ax[0, 2].set_axis_off()
    ax[1, 0].imshow(np.abs(np.fft.fftshift(im_freq_old)),
                    cmap="gray", norm=SymLogNorm(1))
    ax[1, 1].imshow(np.abs(np.fft.fftshift(im_freq)),
                    cmap="gray", norm=SymLogNorm(1))

    ### END YOUR CODE HERE ###
    utils.save_im("moon_filtered.png", utils.normalize(im_filtered))
    plt.show()
