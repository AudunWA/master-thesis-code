from keras_segmentation.data_utils.data_loader import get_segmentation_array
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from prediction_pipeline.depth_modifications.helpers import get_triplets_from_paths, get_depth_array
from prediction_pipeline.depth_modifications.unet_depth_segm import unet_model_from_checkpoint_path

def depth_accuracy(depth_pr, depth_gt, threshold=1.25):
    pr_over_gt = np.nan_to_num(np.divide(depth_pr, depth_gt))
    gt_over_pr = np.nan_to_num(np.divide(depth_gt, depth_pr))
    depth_acc = np.max([pr_over_gt, gt_over_pr], axis=0) < threshold
    return np.mean(depth_acc)


MIN_DEPTH = 1e-3
MAX_DEPTH = 80

def evaluate_depth_segm(model=None, inp_images_dir=None, segm_annotations_dir=None, depth_annotations_dir=None,
             checkpoints_path=None):
    if model is None:
        assert (checkpoints_path is not None), "Please provide the model or the checkpoints_path"
        model = unet_model_from_checkpoint_path(checkpoints_path)

    paths = get_triplets_from_paths(inp_images_dir, segm_annotations_dir, depth_annotations_dir)
    paths = list(zip(*paths))
    inp_images = list(paths[0])
    segm_annotations = list(paths[1])
    depth_annotations = list(paths[2])

    assert type(inp_images) is list
    assert type(segm_annotations) is list
    assert type(depth_annotations) is list
    tp = np.zeros(model.n_classes)
    fp = np.zeros(model.n_classes)
    fn = np.zeros(model.n_classes)

    mean_abs_err_arr = []
    depth_acc_125_arr = []
    depth_acc_125_2_arr = []
    depth_acc_125_3_arr = []

    n_pixels = np.zeros(model.n_classes)

    for inp, segm_ann, depth_ann in tqdm(zip(inp_images, segm_annotations, depth_annotations)):
        segm_pr, depth_pr = model.predict_segmentation(inp)

        segm_gt = get_segmentation_array(segm_ann, model.n_classes, model.output_width, model.output_height, no_reshape=True)

        segm_gt = segm_gt.argmax(-1)


        segm_pr = segm_pr.flatten()
        segm_gt = segm_gt.flatten()



        depth_gt = get_depth_array(depth_ann, model.output_width, model.output_height)

        depth_pr = depth_pr.flatten()
        depth_gt = depth_gt.flatten()
        depth_pr = 1 - depth_pr

        depth_pr[depth_pr > MAX_DEPTH] = MAX_DEPTH
        depth_pr[depth_pr < MIN_DEPTH] = MIN_DEPTH
        depth_gt[depth_gt > MAX_DEPTH] = MAX_DEPTH
        depth_gt[depth_gt < MIN_DEPTH] = MIN_DEPTH


        depth_acc_125 = depth_accuracy(depth_pr, depth_gt, threshold=1.25)
        depth_acc_125_2 = depth_accuracy(depth_pr, depth_gt, threshold=1.25**2)
        depth_acc_125_3 = depth_accuracy(depth_pr, depth_gt, threshold=1.25**3)

        depth_acc_125_arr.append(np.mean(depth_acc_125))
        depth_acc_125_2_arr.append(np.mean(depth_acc_125_2))
        depth_acc_125_3_arr.append(np.mean(depth_acc_125_3))
        
        mean_abs_err = metrics.mean_absolute_error(depth_pr, depth_gt)
        mean_abs_err_arr.append(mean_abs_err)


        for cl_i in range(model.n_classes):
            tp[cl_i] += np.sum((segm_pr == cl_i) * (segm_gt == cl_i))
            fp[cl_i] += np.sum((segm_pr == cl_i) * ((segm_gt != cl_i)))
            fn[cl_i] += np.sum((segm_pr != cl_i) * ((segm_gt == cl_i)))
            n_pixels[cl_i] += np.sum(segm_gt == cl_i)

    mean_abs_err = np.mean(np.array(mean_abs_err_arr))
    depth_acc_125 = np.mean(np.array(depth_acc_125_arr))
    depth_acc_125_2 = np.mean(np.array(depth_acc_125_2_arr))
    depth_acc_125_3 = np.mean(np.array(depth_acc_125_3_arr))
    cl_wise_score = tp / (tp + fp + fn + 0.000000000001)
    n_pixels_norm = n_pixels / np.sum(n_pixels)
    frequency_weighted_IU = np.sum(cl_wise_score * n_pixels_norm)
    mean_IU = np.mean(cl_wise_score)
    return {"frequency_weighted_IU": frequency_weighted_IU, "mean_IU": mean_IU, "class_wise_IU": cl_wise_score, "depth_acc_125": depth_acc_125 , "depth_acc_125_2": depth_acc_125_2,  "depth_acc_125_3": depth_acc_125_3 , "mean_abs_err": mean_abs_err}

