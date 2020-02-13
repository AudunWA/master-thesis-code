import os
from pathlib import Path
import numpy as np
import cv2
from prediction_pipeline.utils.helpers import crop_and_resize, verify_folder_exists
import shutil

# Drivable road
road = np.array([128, 64, 128], dtype='uint8')
man_hole = np.array([100, 128, 160], dtype='uint8')
pot_hole = np.array([70, 100, 150], dtype='uint8')
catch_basin = np.array([220, 128, 128], dtype='uint8')

drivable_road_colors = np.array([road, man_hole, pot_hole, catch_basin])


# Non-drivable road
side_walk = np.array([244, 35, 232], dtype='uint8')
parking = np.array([250, 170, 160], dtype='uint8')
bike_lane = np.array([128, 64, 255], dtype='uint8')
pedestrian_area = np.array([96, 96, 96], dtype='uint8')
service_lane = np.array([110, 110, 110], dtype='uint8')

non_drivable_road_colors = np.array([side_walk, parking, bike_lane, pedestrian_area, service_lane])


# Crosswalk
crosswalk_plain = np.array([140, 140, 200], dtype='uint8')
lane_marking_crosswalk = np.array([200, 128, 128], dtype='uint8')

crosswalk_colors = np.array([crosswalk_plain, lane_marking_crosswalk])


# Lane marking
lane_markings_general = np.array([255,255,255], dtype='uint8')

lane_marking_colors = np.array([lane_markings_general])


# Humans
human = np.array([220, 20, 60], dtype='uint8')
bicyclist = np.array([255, 0, 0], dtype='uint8')
motorcyclist = np.array([255, 0, 100], dtype='uint8')
other_rider = np.array([255, 0, 200], dtype='uint8')

human_colors = np.array([human, bicyclist, motorcyclist, other_rider])


# Traffic light
traffic_light = np.array([250, 170, 30], dtype='uint8')

traffic_light_colors = np.array([traffic_light])


# Vehicles
bicycle = np.array([119, 11, 32], dtype='uint8')
bus = np.array([0, 60, 100], dtype='uint8')
car = np.array([0, 0, 142], dtype='uint8')
caravan = np.array([0, 0, 90], dtype='uint8')
motorcycle = np.array([0, 0, 230], dtype='uint8')
on_rails = np.array([0, 80, 100], dtype='uint8')
other_vehicle = np.array([128, 64, 64], dtype='uint8')
trailer = np.array([0, 0, 110], dtype='uint8')
truck = np.array([0, 0, 80], dtype='uint8')
wheeled_slow = np.array([0, 0, 192], dtype='uint8')

vehicle_colors = np.array([bicycle, bus, car, caravan, motorcycle, on_rails, other_vehicle, trailer, truck, wheeled_slow])

eight_classes = np.array([drivable_road_colors, non_drivable_road_colors, crosswalk_colors, lane_marking_colors, human_colors, traffic_light_colors, vehicle_colors])

three_classes = np.array([drivable_road_colors, lane_marking_colors])

data_folder = Path("data/segmentation_data")


def convert_segmentation_images(input_path, output_path):

    filenames = os.listdir(str(data_folder / input_path))

    output_folder = str(data_folder / output_path)
    verify_folder_exists(output_folder)

    i = 0
    for filename in filenames:

        # In BGR format
        img = cv2.imread(str(data_folder / input_path / filename))
        output_img = np.zeros(img.shape)

        # For each pixel:
        #  If pixel.color in road colors: set pixel to (0,0,0)
        #  If pixel.color in lane marking colors: set pixel to (1,0,0)
        #  Else: set pixel to (2,0,0)

        for j, class_colors in enumerate(three_classes):
            for color in class_colors:
                mask = (img == color[::-1]).all(axis=2)
                output_img[mask] = [j+1, 0, 0]

        write_path = str(Path(data_folder) / output_path / filename)

        cv2.imwrite(write_path, output_img)

        i += 1
        if i % 100 == 0:
            print("Progress: ", i, " of ", len(filenames))




# Prepare original images for train and validation
crop_and_resize("/home/audun/Downloads/mapillary-vistas-dataset_public_v1.1/training/images/", data_folder / "images_prepped_train")
crop_and_resize("/home/audun/Downloads/mapillary-vistas-dataset_public_v1.1/validation/images/", data_folder / "images_prepped_val")

# Prepare original segmentations for further processing

train_cropped_segmentations = data_folder / "output_cropped_train"
validation_cropped_segmentations = data_folder / "output_cropped_val"

crop_and_resize("/home/audun/Downloads/mapillary-vistas-dataset_public_v1.1/training/labels/", train_cropped_segmentations)
crop_and_resize("/home/audun/Downloads/mapillary-vistas-dataset_public_v1.1/validation/labels/", validation_cropped_segmentations)

# Generate dataset with correct number of reduced classes

convert_segmentation_images("output_cropped_train", "annotations_prepped_train")
shutil.rmtree(train_cropped_segmentations)
convert_segmentation_images("output_cropped_val", "annotations_prepped_val")
shutil.rmtree(validation_cropped_segmentations)
