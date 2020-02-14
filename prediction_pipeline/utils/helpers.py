from pathlib import Path
import os
import cv2
from multiprocessing import Process, Value
from multiprocessing.dummy import Pool
import subprocess
from ctypes import c_int
import time

def read_and_crop_image(input_dir, output_dir, filename, counter):

    file_path = input_dir + filename
    write_path = os.path.join(output_dir, filename)

    img = cv2.imread(file_path)

    cv2.imwrite(write_path, crop_and_resize_img(img))
    if counter % 500 == 0:
        print("Progress:", counter)

def crop_and_resize(input_dir, output_dir):
    verify_folder_exists(Path(output_dir))
    #print(" ".join(["parallel", "mogrify", "-path", str(output_dir), "-resize", "256x256\!", ":::", str(input_dir) + "*.(jpg|png)"]))
    #subprocess.run(["parallel", "mogrify", "-path", str(output_dir), "-resize", "256x256\!", ":::", str(input_dir) + "*.(jpg|png)"])
    filenames = list(os.listdir(input_dir))

    with Pool(processes=8) as pool: #this should be the same as your processor cores (or less)
        chunksize = 56 #making this larger might improve speed (less important the longer a single function call takes)

        result = pool.starmap_async(read_and_crop_image, #function to send to the worker pool
                                    ((input_dir, output_dir, file, i) for i, file in enumerate(filenames)),  #generator to fill in function args
                                    chunksize) #how many jobs to submit to each worker at once
        while not result.ready(): #print out progress to indicate program is still working.
            #with counter.get_lock(): #you could lock here but you're not modifying the value, so nothing bad will happen if a write occurs simultaneously
            #just don't `time.sleep()` while you're holding the lock
            time.sleep(.5)
        print('\nCompleted all images')
    """
    filenames = list(os.listdir(input_dir))
    filenames.sort()
    filename_chunks = [filenames[i:i + 10] for i in range(0, len(filenames), 10)]
    print("Number of images", len(filenames))
    workers = []
    for filename_chunk in filename_chunks:
        worker = Process(target=read_and_crop_image, args=(input_dir, output_dir, filename_chunk))
        worker.start()
        workers.append(worker)

    for worker in workers:
        worker.join()
    """


def crop_and_resize_img(img):
    side_len = min(img.shape[0], img.shape[1])
    side_len -= side_len % 32
    cropped_img = img[0:side_len, 0:side_len]
    return cv2.resize(cropped_img, (256, 256), interpolation=cv2.INTER_NEAREST)

def verify_folder_exists(path):
    if not os.path.exists(str(path)):
        os.makedirs(str(path))
