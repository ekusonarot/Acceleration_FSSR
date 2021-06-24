import numpy as np
import cv2
import torch

class Converter(): 
    def convert_ycbcr_to_bgr(img):
        r = 298.082 * img[..., 0] / 256. + 408.583 * img[..., 2] / 256. - 222.921
        g = 298.082 * img[..., 0] / 256. - 100.291 * img[..., 1] / 256. - 208.120 * img[..., 2] / 256. + 135.576
        b = 298.082 * img[..., 0] / 256. + 516.412 * img[..., 1] / 256. - 276.836
        return np.array([b, g, r]).transpose([1, 2, 0])

    def convert_bgr_to_ycbcr(img):
        y = 16. + (64.738 * img[..., 2] + 129.057 * img[..., 1] + 25.064 * img[..., 0]) / 256.
        cb = 128. + (-37.945 * img[..., 2] - 74.494 * img[..., 1] + 112.439 * img[..., 0]) / 256.
        cr = 128. + (112.439 * img[..., 2] - 94.154 * img[..., 1] - 18.285 * img[..., 0]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])

    def convert_bgr_to_y(img):
        return (16. + (64.738 * img[..., 2] + 129.057 * img[..., 1] + 25.064 * img[..., 0]) / 256.) / 255.

def LPIPSpreprocess(image, factor=255./2., cent=1.):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    image = image / factor - cent
    image = image[np.newaxis]
    return torch.from_numpy(image.astype(np.float32))