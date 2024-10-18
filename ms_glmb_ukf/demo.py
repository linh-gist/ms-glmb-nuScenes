import os
import pandas as pd

from gen_model import model
from gen_meas import load_detection
from run_filter import GLMB
from clearmot import clearmot_single_dataset, clear_mot
import multiprocessing
import numpy as np


def dataset_eval(dataset, adaptive_birth, use_feat, exp_idx="", root_dir="./results"):
    model_params = model(dataset)
    meas, img_dirs = load_detection(model_params, dataset)
    glmb = GLMB(model_params, adaptive_birth, use_feat)
    glmb.run(model_params, dataset, meas)
    # glmb.runcpp(model_params, dataset, meas, adaptive_birth, use_feat)
    glmb.save_est_motformat(root_dir, dataset + str(exp_idx) + "_" + str(adaptive_birth) + "_" + str(use_feat))


def run_tests(dataset, start, end, adaptive_birth, use_feat):
    root_results_dir = "./results"
    gt_data_dir = "../../data/images/"
    processes = []
    for exp_idx in range(start, end):
        p = multiprocessing.Process(target=dataset_eval, args=(dataset, adaptive_birth, use_feat, exp_idx))
        processes.append(p)
        p.start()
    print("Waiting all processes to be finished................")
    for process in processes:
        process.join()
    gt_list = []
    est_list = []
    dataset_list = []
    for exp_idx in range(start, end):
        exp_result = dataset + str(exp_idx) + "_" + str(adaptive_birth) + "_" + str(use_feat)
        gt_file = os.path.join(gt_data_dir, dataset, "GT_" + dataset + "_WORLD_CENTROID.txt")
        np_gt = pd.read_csv(gt_file, delimiter=' ', header=None).to_numpy()
        est_file = os.path.join(root_results_dir, "EST_" + exp_result + "_WORLD_CENTROID.txt")
        est = pd.read_csv(est_file, delimiter=' ', header=None).to_numpy()
        gt_list.append(np_gt)
        est_list.append(est)
        dataset_list.append(exp_result)
    ospa2_strsummary = ospa2_datasets(gt_list, est_list, dataset_list)
    print(ospa2_strsummary)
    strsummary = clear_mot(gt_list, est_list, dataset_list)
    print(strsummary)
    with open(os.path.join(root_results_dir, 'summary_clearmot.txt'), 'w') as f:
        f.write(strsummary)


if __name__ == '__main__':
    results = {}
    import json

    f1 = open("./nuscenes/scenes_dict_val.json")
    json_data = json.load(f1)
    f2 = open("../detection/lidar/centerpoint/val_centerpoint_rearrange.json")
    lidar_dets = json.load(f2)

    scenes = list(json_data.keys())
    for idx_scene, dataset in enumerate(scenes):
        # if idx_scene < 126:
        #     continue
        # dataset = "CMC1"  # CMC1, CMC2, CMC3, CMC4, CMC5, WILDTRACK
        birth_opt = 3  # (0[Fix birth], 1[Monte Carlo (AB)], 2[KMeans (AB)], 3[MeanShift (AB)])
        model_params = model("WILDTRACK")
        model_params.N_sensors = 6
        centers = []
        for key in json_data.keys():
            for frame in json_data[key].keys():
                boxes = json_data[key][frame]["boxes"]
                for box in boxes:
                    centers.append(box["center"])
        centers = np.column_stack(centers)
        min = np.min(centers, axis=1)
        max = np.max(centers, axis=1)
        offset = 20
        model_params.XMAX = [min[0] - offset, max[0] + offset]
        model_params.YMAX = [min[1] - offset, max[1] + offset]
        model_params.ZMAX = [min[2] - 1, max[2] + 2]
        model_params.pdf_c[6] = np.log(1 / ((max[0] - min[0]) * (max[1] - min[1]) * (max[2] - min[2])))
        yolo_class_names = {0: "pedestrian", 2: "car", 7: "truck", 5: "bus", 6: "trailer", 3: "motorcycle",
                            1: "bicycle"}
        # confs = {0: 0.5, 2: 0.5, 7: 0.4, 5: 0.4, 6: 0.4, 3: 0.3, 1: 0.3}
        conf = 0.45
        for class_id in yolo_class_names.keys():
            print(idx_scene, dataset, yolo_class_names[class_id], "=========================================")
            # if yolo_class_names[class_id] != "bus":
            #     continue
            if idx_scene > 2:
                continue
            meas, img_dirs = load_detection(model_params, json_data[dataset], dataset, class_id)

            # Choose the following options (adaptive_birth = 0 [Fix birth], 1 [Monte Carlo], 2 [MeanShift])
            # (0) Fix birth uses re-id feature => [adaptive_birth=0, use_feat=False] => ONLY for CMC dataset
            # (1.1) Monte Carlo Adaptive birth uses re-id feature [adaptive_birth=1, use_feat=True]
            # (1.2) Monte Carlo Adaptive birth does NOT use re-id feature [adaptive_birth=1, use_feat=False]
            # (2.1) KMeans Adaptive birth uses re-id feature [adaptive_birth=2, use_feat=True]
            # (2.2) KMeans Adaptive birth does NOT use re-id feature [adaptive_birth=2, use_feat=False]
            # (3.1) MeanShift Adaptive birth uses re-id feature [adaptive_birth=2, use_feat=True]
            # (3.2) MeanShift Adaptive birth does NOT use re-id feature [adaptive_birth=2, use_feat=False]
            glmb = GLMB(model_params, adaptive_birth=3, use_feat=False)
            # est_params = glmb.run(model_params, dataset + "_" + str(class_id), meas, json_data[dataset],
            #                       lidar_dets[dataset], class_id - 1)
            est_params = glmb.runcpp(model_params, dataset + "_" + yolo_class_names[class_id], meas, json_data[dataset],
                                     lidar_dets[dataset], yolo_class_names[class_id], class_id, conf,
                                     adaptive_birth=birth_opt, use_feat=False)
            # clearmot_single_dataset(est_params, dataset)
            # ospa2_single_dataset(est_params, dataset)
            # plot_3d_video(model_params, est_params)
            # making_demo_video(model_params, dataset, est_params)
f1.close()
f2.close()
# Running repeated experiments parallelly
# datasets = ["CMC1", "CMC2", "CMC3", "CMC4", "CMC5", "WILDTRACK"]
# start, end = 0, 25
# for dataset in datasets:
#     birth_opt = 1
#     run_tests(dataset, start, end, adaptive_birth=birth_opt, use_feat=False)
#     run_tests(dataset, start, end, adaptive_birth=birth_opt, use_feat=True)
#
#     birth_opt = 2
#     run_tests(dataset, start, end, adaptive_birth=birth_opt, use_feat=False)
#     run_tests(dataset, start, end, adaptive_birth=birth_opt, use_feat=True)
#
#     birth_opt = 3
#     run_tests(dataset, start, end, adaptive_birth=birth_opt, use_feat=False)
#     run_tests(dataset, start, end, adaptive_birth=birth_opt, use_feat=True)
