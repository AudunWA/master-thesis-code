from pathlib import Path
import os
import cv2
from multiprocessing.dummy import Pool
import time
import numpy as np

def read_and_crop_image(input_dir, output_dir, filename, counter):
    try:
        file_path = input_dir + filename
        write_path = os.path.join(output_dir, filename)

        if ".npz" in filename:
            try:
                img = np.load(file_path)["arr_0"]
                img = img.reshape((img.shape[0], img.shape[1], 1)) * 255 * 3 # 1 Channel
                img = img.astype("uint8")
            except Exception as e:
                print("Exception:", e)
                raise e
        else:
            img = cv2.imread(file_path)
        try:
            cv2.imwrite(write_path.replace("npz", "jpg"), crop_and_resize_img(img))
        except Exception as e:
            print("Exception on write:", e)
            raise e
        if counter % 500 == 0:
            print("Progress:", counter)
    except Exception as e:
        print("Exception:", e)

def crop_and_resize(input_dir, output_dir):
    verify_folder_exists(Path(output_dir))
    filenames = list(os.listdir(input_dir))
    print("output_dir", output_dir)

    print("Processing {} images".format(len(filenames)))
    with Pool(processes=8) as pool: #this should be the same as your processor cores (or less)
        chunksize = 56 #making this larger might improve speed (less important the longer a single function call takes)
        print("chunksize", chunksize)

        result = pool.starmap_async(read_and_crop_image, #function to send to the worker pool
                                    ((input_dir, output_dir, file, i) for i, file in enumerate(filenames)),  #generator to fill in function args
                                    chunksize) #how many jobs to submit to each worker at once
        while not result.ready(): #print out progress to indicate program is still working.
            #with counter.get_lock(): #you could lock here but you're not modifying the value, so nothing bad will happen if a write occurs simultaneously
            #just don't `time.sleep()` while you're holding the lock

            time.sleep(.1)
        print('\nCompleted all images')


def crop_and_resize_img(img):
    side_len = min(img.shape[0], img.shape[1])
    side_len -= side_len % 32
    cropped_img = img[0:side_len, 0:side_len]
    return cv2.resize(cropped_img, (256, 256), interpolation=cv2.INTER_NEAREST)

def verify_folder_exists(path):
    if not os.path.exists(str(path)):
        os.makedirs(str(path))
