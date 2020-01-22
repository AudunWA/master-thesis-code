import os
import numpy as np
import cv2
from skimage import io, color


road_color = np.array([128, 64, 128], dtype='uint8')
lane_markings = np.array([196, 196, 196], dtype='uint8')


for filename in os.listdir("./data/output"):

    output_img = cv2.imread("./data/output/" + filename)

    mask = np.array(output_img == road_color, dtype='uint8') * 255

    road_or_not_mask = np.all(mask, axis=2)

    # RGB Image with B as class indices
    output_format = np.zeros((road_or_not_mask.shape[0], road_or_not_mask.shape[1], 3))

    output_format[road_or_not_mask] = [1, 0, 0]

    cv2.imwrite("./data/results/" + filename, output_format)
