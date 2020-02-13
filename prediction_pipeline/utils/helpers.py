from pathlib import Path
import os
import cv2

def crop_and_resize(input_dir, output_dir):
    verify_folder_exists(Path(output_dir))
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
    return cv2.resize(cropped_img, (256, 256), interpolation=cv2.INTER_NEAREST)

def verify_folder_exists(path):
    if not os.path.exists(str(path)):
        os.makedirs(str(path))
