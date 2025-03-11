import cv2 as cv
import numpy as np

for i in range(90):
    filepath = './public_data/lego_full/image.old/{:03d}.png'.format(i)
    img = cv.imread(filepath, flags=cv.IMREAD_UNCHANGED)
    img = img.astype(np.float32)
    img = img / 255.0
    # extract alpha path
    alpha = img[:, :, 3:]
    color = img[:, :, :3]
    blended = alpha * color + np.zeros_like(color) * (1. - alpha)
    blended = blended * 255.0
    blended = blended.astype(np.uint8)
    cv.imwrite('./public_data/lego_full/image/{:03d}.png'.format(i), blended)
