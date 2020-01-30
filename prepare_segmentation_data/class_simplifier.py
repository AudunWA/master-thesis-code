import os
import random

import numpy as np
import cv2
from skimage import io, color

def crop_and_resize(input_dir, output_dir):
    filenames = list(os.listdir(input_dir))
    filenames.sort()
    print("Number of images", len(filenames))
    i = 0
    for filename in filenames:
        file_path = input_dir + filename
        write_path = os.path.join(output_dir, filename)

        img = cv2.imread(file_path)
        cv2.imwrite(write_path, crop_and_resize_img(img))
        i += 1
        if i % 500 == 0:
            print("Progress: ", i, " of ", len(filenames))


def crop_and_resize_img(img):
    side_len = min(img.shape[0], img.shape[1])
    side_len -= side_len % 32
    cropped_img = img[0:side_len, 0:side_len]
    return cv2.resize(cropped_img, (512, 512), interpolation=cv2.INTER_NEAREST)


#crop_and_resize("../../training/images/", "data/dataset2/images_prepped_train")
#crop_and_resize("../../training/images/", "data/dataset2/images_prepped_test")

#crop_and_resize("../../training/labels/", "data/output_cropped")


road = np.array([128, 64, 128], dtype='uint8') # Road
side_walk = np.array([244, 35, 232], dtype='uint8') # Road
parking = np.array([250, 170, 160], dtype='uint8') # Road
service_lane = np.array([110, 110, 110], dtype='uint8') # Road
bike_lane = np.array([128, 64, 255], dtype='uint8') # Road
pedestrian_area = np.array([96, 96, 96], dtype='uint8') # Road

cross_walk_plain = np.array([140, 140, 200], dtype='uint8') # Road
lane_marking_crosswalk = np.array([200, 128, 128], dtype='uint8') # Road
lane_markings_general = np.array([255,255,255], dtype='uint8') #Lane marking

man_hole = np.array([100, 128, 160], dtype='uint8')# Road
pot_hole = np.array([70, 100, 150], dtype='uint8')# Road
catch_basin = np.array([220, 128, 128], dtype='uint8')# Road

road_colors = np.array([road, side_walk, parking, service_lane, bike_lane, pedestrian_area, cross_walk_plain, lane_marking_crosswalk, man_hole, pot_hole, catch_basin])
lane_marking_colors = np.array([lane_markings_general])

i = 0
filenames = os.listdir("./data/output_cropped")
for filename in filenames:

    # In BGR format
    img = cv2.imread("./data/output_cropped/" + filename)
    output_img = np.zeros(img.shape)

    # For each pixel:
    #  If pixel.color in road colors: set pixel to (0,0,0)
    #  If pixel.color in lane marking colors: set pixel to (1,0,0)
    #  Else: set pixel to (2,0,0)
    for color in road_colors:
        mask = (img == color[::-1]).all(axis=2)
        output_img[mask] = [1, 0, 0]

    for color in lane_marking_colors:
        mask = (img == color[::-1]).all(axis=2)
        output_img[mask] = [2, 0, 0]

    write_path = os.path.join("./data/dataset2/annotations_prepped_train", filename)
    cv2.imwrite(write_path, output_img)

    i += 1
    if i % 100 == 0:
        print("Progress: ", i, " of ", len(filenames))

    """ mask = np.array(output_img == road_color, dtype='uint8') * 255

    road_or_not_mask = np.all(mask, axis=2)

    # RGB Image with B as class indices
    output_format = np.zeros((road_or_not_mask.shape[0], road_or_not_mask.shape[1], 3))

    output_format[road_or_not_mask] = [1, 0, 0]

    write_path = os.path.join("./data/dataset2/annotations_prepped_train", filename)

    cv2.imwrite(write_path, output_format)

    #write_path = os.path.join("./data/dataset2/annotations_prepped_test", filename)
"
    #cv2.imwrite(write_path, output_format)"""


