import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

import numpy as np
import skimage
import utils


def convolve_im(im: np.array,
                fft_kernel: np.array,
                verbose=True):
    """ Convolves the image (im) with the frequency kernel (fft_kernel),
        and returns the resulting image.

        "verbose" can be used for turning on/off visualization
        convolution

    Args:
        im: np.array of shape [H, W]
        fft_kernel: np.array of shape [H, W]
        verbose: bool
    Returns:
        im: np.array of shape [H, W]
    """
    # START YOUR CODE HERE ### (You can change anything inside this block)
    fft = np.fft.fft2(im)
    prod = fft * fft_kernel
    conv_result = np.abs(np.fft.ifft2(prod))
    if verbose:
        # Use plt.subplot to place two or more images beside eachother
        plt.figure(figsize=(20, 4))
        # plt.subplot(num_rows, num_cols, position (1-indexed))
        plt.subplot(2, 3, 1)
        plt.imshow(im, cmap="gray")
        plt.subplot(2, 3, 2)
        plt.imshow(np.abs(np.fft.fftshift(fft)),
                   cmap="gray", norm=SymLogNorm(1))
        plt.subplot(2, 3, 4)
        plt.imshow(np.abs(np.fft.fftshift(fft_kernel)),
                   cmap="gray", norm=SymLogNorm(1))
        plt.subplot(2, 3, 5)
        plt.imshow(np.abs(np.fft.fftshift(prod)),
                   cmap="gray", norm=SymLogNorm(1e-3))
        plt.subplot(2, 3, 6)
        plt.imshow(conv_result, cmap="gray")

    ### END YOUR CODE HERE ###
    return conv_result


if __name__ == "__main__":
    verbose = True
    # Changing this code should not be needed
    im = skimage.data.camera()
    im = utils.uint8_to_float(im)
    # DO NOT CHANGE
    frequency_kernel_low_pass = utils.create_low_pass_frequency_kernel(
        im, radius=50)
    image_low_pass = convolve_im(im, frequency_kernel_low_pass,
                                 verbose=verbose)
    # DO NOT CHANGE
    frequency_kernel_high_pass = utils.create_high_pass_frequency_kernel(
        im, radius=50)
    image_high_pass = convolve_im(im, frequency_kernel_high_pass,
                                  verbose=verbose)

    if verbose:
        plt.show()
    utils.save_im("camera_low_pass.png", image_low_pass)
    utils.save_im("camera_high_pass.png", image_high_pass)
