import numpy
import math

def psnr(img1, img2):
    '''Calculate peak signal to noise ratio (PSNR)
    See https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio '''
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))