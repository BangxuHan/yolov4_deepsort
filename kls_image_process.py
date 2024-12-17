import cv2
import numpy as np


def cv2_letterbox_image(image, expected_size):
    ih, iw = image.shape[0:2]
    ew, eh = expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    new_img = np.zeros((eh, ew, 3), dtype=np.uint8)

    top = (eh - nh) // 2
    bottom = nh + top
    left = (ew - nw) // 2
    right = nw + left
    new_img[top:bottom, left:right, :] = image
    # new_img = new_img / 255.
    # new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return new_img
