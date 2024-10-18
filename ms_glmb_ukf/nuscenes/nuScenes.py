# %matplotlib inline
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
import cv2
import json
import numpy as np
import pandas as pd
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.data_classes import TrackingConfig, TrackingBox
from nuscenes.eval.tracking.evaluate import TrackingEval, load_gt, add_center_dist, filter_eval_boxes, create_tracks
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox


def compute_proj_mat(pose_record, cs_record):
    # (cs translation, cs rotation, camera_intrinsic) seem to be identical across frames in the same scene
    # (pose translation, pose rotation) are not equal in different frames
    t1 = -np.array(pose_record['translation'])
    pose_rotation = pose_record['rotation']
    R1 = Quaternion(pose_rotation).inverse.rotation_matrix
    # R (x + t) = R x + t'. You want this t' ? If so, it will be t' = R t .
    t1 = np.dot(R1, t1)
    E1 = np.column_stack((R1, t1))

    t2 = -np.array(cs_record['translation'])
    cs_rotation = cs_record['rotation']
    R2 = Quaternion(cs_rotation).inverse.rotation_matrix
    # R (x + t) = R x + t'. You want this t' ? If so, it will be t' = R t .
    t2 = np.dot(R2, t2)
    E2 = np.column_stack((R2, t2))
    cs_camera_intrinsic = cs_record['camera_intrinsic']

    # The combined transformation matrix (E) can be obtained by multiplying E2 with E1
    E = np.dot(np.row_stack((E2, [0, 0, 0, 1])), np.row_stack((E1, [0, 0, 0, 1])))
    prj_mat = np.dot(cs_camera_intrinsic, E[0:3, :])
    return prj_mat, E[0:3, :]


def get_scene_data2json(nusc, split="val", file='scenes_dict', visual=False):
    splits = create_splits_scenes()
    print(f'These are the splits in nuScenes: {list(splits.keys())}')
    scenes_in_split = list(splits[split])
    scenes_dict = {}
    if visual:
        fig, ax = plt.subplots()

    for scene in nusc.scene:
        scene_name = scene['name']
        if scene_name in scenes_in_split:
            scene_dict = {}
            first_sample_token = scene['first_sample_token']
            last_sample_token = scene['last_sample_token']
            sensors = ["CAM_BACK", "CAM_FRONT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
            frame_index = 0
            current_token = first_sample_token
            keep_looping = True
            while keep_looping:
                files = []
                prj_mats = []
                cam_exts = []
                if current_token == last_sample_token:
                    keep_looping = False
                sample_record = nusc.get('sample', current_token)
                boxes = nusc.get_boxes(nusc.get('sample_data', sample_record['data']["CAM_BACK"])['token'])
                for sensor in sensors:
                    cam_data = nusc.get('sample_data', sample_record['data'][sensor])
                    # rdata = nusc.render_sample_data(cam_front_data['token'])
                    # nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP')
                    rdata = nusc.get_sample_data(cam_data['token'])  # data_path, box_list, cam_intrinsic
                    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
                    pose_record = nusc.get('ego_pose', cam_data['ego_pose_token'])
                    prj_mat, cam_ext = compute_proj_mat(pose_record, cs_record)
                    prj_mats.append(prj_mat.tolist())
                    cam_exts.append(cam_ext.tolist())
                    files.append(rdata[0])
                    if visual and sensor == sensors[0]:  # Only show the first camera
                        box_vis_level = BoxVisibility.ALL
                        data_path, boxes, cs_camera_intrinsic = nusc.get_sample_data(cam_data['token'],
                                                                                     box_vis_level=box_vis_level)
                        boxes_3d = nusc.get_boxes(cam_data['token'])
                        img = Image.open(data_path)
                        ax.imshow(img)
                        for box in boxes_3d:
                            cc = np.copy(box.center)
                            vet2 = np.array([[cc[0]], [cc[1]], [cc[2]], [1]])
                            temp_c = np.dot(prj_mat, vet2)
                            point_3d2img = temp_c[0:2] / temp_c[2]
                            imsize = (cam_data['width'], cam_data['height'])
                            # Move box to ego vehicle coord system.
                            box.translate(-np.array(pose_record['translation']))
                            box.rotate(Quaternion(pose_record['rotation']).inverse)
                            #  Move box to sensor coord system.
                            box.translate(-np.array(cs_record['translation']))
                            box.rotate(Quaternion(cs_record['rotation']).inverse)
                            check = box_in_image(box, cs_camera_intrinsic, imsize, vis_level=box_vis_level)
                            if not check:
                                continue
                            print(scene_name, frame_index, point_3d2img)
                            plt.plot(point_3d2img[0], point_3d2img[1], marker='v', color="red")
                        for box in boxes:
                            c = np.array(nusc.colormap[box.name]) / 255.0
                            box.render(ax, view=cs_camera_intrinsic, normalize=True, colors=(c, c, c))
                        ax.set_xlim(0, img.size[0])
                        ax.set_ylim(img.size[1], 0)
                        # plt.show()
                        fig.canvas.draw()  # redraw the canvas
                        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)  # convert canvas to image
                        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.imshow(scene_name + "_3D", img)
                        ax.cla()
                        img0 = cv2.imread(data_path)  # cv2.imread(rdata[0])
                        cv2.imshow(scene_name, img0)
                        cv2.waitKey(0)
                next_token = sample_record['next']
                current_token = next_token
                tmp_boxes = []
                for box in boxes:
                    box_instance_token = nusc.get('sample_annotation', box.token)["instance_token"]
                    tmp_boxes.append({"center": box.center.tolist(), "token": box.token, "wlh": box.wlh.tolist(),
                                      "name": box.name, "rotation_matrix": box.rotation_matrix.tolist(),
                                      "orientation": box.orientation.q.tolist(), "instance_token": box_instance_token})
                scene_dict[frame_index] = {"files": files, "prj_mats": prj_mats, "sample_token": sample_record["token"],
                                           "boxes": tmp_boxes, "cam_exts": cam_exts}
                frame_index = frame_index + 1
            scenes_in_split.remove(scene_name)
            scenes_dict[scene_name] = scene_dict
            print(scene_name, "Total Frames:", frame_index)
    with open(file + "_" + split + ".json", 'w') as fp:
        json.dump(scenes_dict, fp)
    ################


def get_lidar_dets(nusc, split="val", detfile='../../test/detection-megvii/megvii_val.json'):
    tracking_names = ["bicycle", "bus", "car", "motorcycle", "pedestrian", "trailer", "truck"]

    splits = create_splits_scenes()
    print(f'These are the splits in nuScenes: {list(splits.keys())}')
    scenes_in_split = list(splits[split])
    scenes_dict = {}
    fd = open(detfile)
    det_results = EvalBoxes.deserialize(json.load(fd)["results"], DetectionBox)

    for scene in nusc.scene:
        scene_name = scene['name']
        if scene_name in scenes_in_split:
            scene_dict = {}
            first_sample_token = scene['first_sample_token']
            last_sample_token = scene['last_sample_token']
            frame_index = 0
            current_token = first_sample_token
            keep_looping = True
            while keep_looping:
                if current_token == last_sample_token:
                    keep_looping = False
                ###
                tmp_boxes = []
                for box in det_results[current_token]:
                    if box.detection_name not in tracking_names:
                        continue
                    # tmp_boxes.append({"center": box["translation"], "wlh": box["size"], "rotation": box["rotation"],
                    #                   "velocity": box["velocity"], "name": box["detection_name"]})
                    tmp_boxes.append({"center": box.translation, "wlh": box.size, "rotation": box.rotation,
                                      "velocity": box.velocity, "name": box.detection_name,
                                      "score": box.detection_score})
                ###
                sample_record = nusc.get('sample', current_token)
                next_token = sample_record['next']
                current_token = next_token
                scene_dict[frame_index] = {"sample_token": sample_record["token"], "boxes": tmp_boxes}
                frame_index = frame_index + 1
            scenes_in_split.remove(scene_name)
            scenes_dict[scene_name] = scene_dict
            print(scene_name, "Total Frames:", frame_index)
    with open("megvii_det" + split + ".json", 'w') as fp:
        json.dump(scenes_dict, fp)
    fd.close()
    ################


def gen_motchallenge_gt(nusc, eval_set="val", verbose=True):
    # Reference: nuscenes\eval\tracking\evaluate.py TrackingEval
    cfg = config_factory('tracking_nips_2019')
    # Load data.
    gt_boxes = load_gt(nusc, eval_set, TrackingBox, verbose=verbose)
    # Add center distances.
    gt_boxes = add_center_dist(nusc, gt_boxes)
    # Filter boxes (distance, points per box, etc.).
    gt_boxes = filter_eval_boxes(nusc, gt_boxes, cfg.class_range, verbose=verbose)
    # Convert boxes to tracks format.
    tracks_gt = create_tracks(gt_boxes, nusc, eval_set, gt=True)

    splits = create_splits_scenes()
    scenes_in_split = list(splits[eval_set])
    unique_int_id = 0
    scene_idx = 0
    for scene in nusc.scene:
        scene_name = scene['name']
        if scene_name not in scenes_in_split:
            continue
        tracks_in_scenes = tracks_gt[scene["token"]]
        timestamps = sorted(tracks_in_scenes.keys())
        tname_lines = {}
        tracking_ids = {}
        for frame_id, timestamp in enumerate(timestamps):
            for trackingbox in tracks_in_scenes[timestamp]:
                x, y, z = trackingbox.translation
                wx, wy, h = trackingbox.size
                tracking_id = trackingbox.tracking_id
                tname = trackingbox.tracking_name
                if tracking_id in tracking_ids:
                    tt_id = tracking_ids[tracking_id]
                else:
                    tracking_ids[tracking_id] = unique_int_id
                    tt_id = unique_int_id
                    unique_int_id = unique_int_id + 1
                line = [int(frame_id) + 1, tt_id, -1, -1, -1, -1, -1, x, y, z, wx / 2, wy / 2, h / 2]
                if tname not in tname_lines.keys():
                    tname_lines[tname] = []  # Init empty list
                tname_lines[tname].append(line)
        for tname in tname_lines.keys():
            print(scene_idx, scene_name, tname)
            gt_scene = np.array(tname_lines[tname])
            np.savetxt(os.path.join("./gt", 'GT_{}_{}_WORLD_CENTROID.txt'.format(scene_name, tname)), gt_scene)
        scene_idx = scene_idx + 1
        # End of a scene
    ############


def get_color(idx):
    idx = (idx + 1) * 50
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

def homtrans(T, p):
    if T.shape[0] == p.shape[0]:
        if len(p.shape) == 3:
            pt = []
            for i in range(p.shape[2]):
                pt.append(np.dot(T, p[:, :, i]))
            pt = np.stack(pt, axis=2)
        else:
            pt = np.dot(T, p)
    else:
        if T.shape[0] - p.shape[0] == 1:
            e2h = np.row_stack((p, np.ones(p.shape[1])))  # E2H Euclidean to homogeneous
            temp = np.dot(T, e2h)
            # H2E Homogeneous to Euclidean
            numrows = temp.shape[0]
            pt = temp[:numrows - 1, :] / np.tile(temp[numrows - 1, :], (numrows - 1, 1))
        else:
            print("matrices and point data do not conform")
    return pt
def draw_on_images(cam_mat, cam_ext, state_3d, bboxes, img0, k, s, is_bbox3d=False):
    # bboxes : ltrb, state_3d: id, x, y, z, wx, wy, h
    for ibbox, bbox in enumerate(bboxes):
        bbox = np.array([[827.8249   ], [495.4664   ], [  4.56464  ], [  5.4235573]])
        bbox[2:4] = np.exp(bbox[2:4])
        l, t = int(bbox[0]), int(bbox[1])
        r, b = int(l+bbox[2]), int(t+bbox[3])
        # draw bbox
        # img0 = cv2.circle(img0, (l, t), radius=8, color=(255, 255, 255), thickness=-1)
        img0 = cv2.rectangle(img0, (l, t), (r, b), color=(255, 255, 255), thickness=2)
        # img0 = cv2.putText(img0, str(ibbox), org=(l, t), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
        #                    color=(0, 255, 255), thickness=2)
    img0 = cv2.putText(img0, "{}, Frame {}".format(s, k), org=(100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                       fontScale=1.5, color=(0, 255, 255), thickness=2)
    # from ..gen_meas import homtrans
    homtrans(np.linalg.inv(cam_mat[:, [0, 1, 3]]),
             np.array([[298.75803 + np.exp(4.19478) / 2], [487.2112 + np.exp(3.7845912) / 2]]))
    for ibbox, bbox in enumerate(state_3d):
        obj_id = int(state_3d[ibbox, 0])
        cx, cy, cz, rx, ry, rz = state_3d[ibbox, 1:]
        # cx = 2.87720422e+02
        # cy = 9.13315884e+02
        # cz = 1.24989822e+00
        # rx = np.exp(-1.13320805e+00)
        # ry = np.exp(-1.02910228e+00)
        # rz = np.exp(-1.05217709e-01)
        arr = np.array([[ 8.60412360e+02] , [ 1.66633316e+03],   [ 5.02817168e-01],  [-1.17260827e+00], [-1.06128531e+00], [-1.11015514e-01]])
        arr[3:6 ] = np.exp(arr[3:6])
        cx, cy, cz, rx, ry, rz = arr
        point = np.dot(cam_ext, np.array([cx, cy, cz, 1]))
        print(point)
        if point[2] < 0 :
            continue  # Do not draw objects far from cameras

        if is_bbox3d:
            vet = np.ones((4, 8))
            vet[:, 0] = [cx - rx, cy - ry, cz - rz, 1]
            vet[:, 1] = [cx + rx, cy + ry, cz - rz, 1]
            vet[:, 2] = [cx - rx, cy + ry, cz - rz, 1]
            vet[:, 3] = [cx + rx, cy - ry, cz - rz, 1]
            vet[:, 4] = [cx - rx, cy - ry, cz + rz, 1]
            vet[:, 5] = [cx + rx, cy + ry, cz + rz, 1]
            vet[:, 6] = [cx - rx, cy + ry, cz + rz, 1]
            vet[:, 7] = [cx + rx, cy - ry, cz + rz, 1]
            temp = np.dot(cam_mat, vet)
            x_p = (temp[[0, 1], :] / temp[2, :]).astype("int")
            # Define the indices of the points to draw lines between
            indices = [[0, 2], [1, 3], [0, 3], [1, 2],
                       [4, 6], [5, 7], [4, 7], [5, 6],
                       [0, 4], [3, 7], [1, 5], [2, 6]]
            # Draw the lines
            for i, j in indices:
                cv2.line(img0, (x_p[0, i], x_p[1, i]), (x_p[0, j], x_p[1, j]), get_color(obj_id), 2, 2, 0)
            continue
        # generates three (N)-by-(N) matrices so that SURF(X,Y,Z) produces an
        # ellipsoid with center (XC,YC,ZC) and radii XR, YR, ZR
        N = 150
        u = np.linspace(0, 2 * np.pi, N)
        v = np.linspace(0, np.pi, N)
        x = rx * np.outer(np.cos(u), np.sin(v)) + cx
        y = ry * np.outer(np.sin(u), np.sin(v)) + cy
        z = rz * np.outer(np.ones_like(u), np.cos(v)) + cz
        # projection 3d ellipsoid points to image plane
        vet = np.ones((4, N ** 2))  # (x, y, z, 1)
        for idx in range(N):
            tempp = np.row_stack((x[:, idx], y[:, idx], z[:, idx]))
            vet[:3, idx * N:(idx + 1) * N] = tempp
        temp = np.dot(cam_mat, vet)
        img1_vert = (temp[[0, 1], :] / temp[2, :]).astype("int")

        i_ty = np.argmin(img1_vert[1, :])
        tx, ty = (img1_vert[0, i_ty], img1_vert[1, i_ty])
        img0 = cv2.putText(img0, str(obj_id), org=(tx, ty), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                           fontScale=0.8, color=(0, 255, 255), thickness=2)

        # Get the indices of the pixels to be modified
        row_indices, col_indices = img1_vert[1, :], img1_vert[0, :]
        # Use boolean indexing to select the pixels that are within the bounds of the image
        valid_rows = (row_indices >= 0) & (row_indices < img0.shape[0])
        valid_cols = (col_indices >= 0) & (col_indices < img0.shape[1])
        valid_pixels = valid_rows & valid_cols
        # Set the values of the valid pixels to 255
        img0[row_indices[valid_pixels], col_indices[valid_pixels], :] = get_color(obj_id)
    # cv2.imwrite("./results/" + str(k) + "_" + str(s) + ".jpg", img0)
    # cv2.imshow("Image", img0)
    # cv2.waitKey(0)
    return img0


def visual_tracking_results(data_file="scenes_dict_val.json", result_path="../results/nuscenes"):
    store_data_dir = "../results/nuscenes/"
    bbxs_det_dir = "../../detection/fairmotx/nuscenes/"
    fd = open(data_file)
    json_data = json.load(fd)
    scenes = list(json_data.keys())
    sensors = ["CAM_BACK", "CAM_FRONT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
    for idx, scene in enumerate(scenes):
        print(idx, scene)
        if idx!=5:
            continue
        size = (1600, 900)
        scale_percent = 0.5  # percent of original size
        video_size = (size[0] * 3, size[1] * 2)
        scaled_size = (int(video_size[0] * scale_percent), int(video_size[1] * scale_percent))
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # openh264-1.7.0-win64.dll
        out = cv2.VideoWriter(store_data_dir + scene + ".avi", fourcc, 2, scaled_size)
        bbxs = [np.load(os.path.join(bbxs_det_dir, scene, "Cam_{}.npz".format(s))) for s in range(1, 7)]
        for scene_frame in json_data[scene].keys():
            ca_data = []
            if scene_frame!=str(18):
                continue
            for class_id in range(1, 9):
                tmp_file = 'EST_{}_WORLD_CENTROID.txt'.format(scene + "_" + str(class_id))
                tmp = np.loadtxt(os.path.join(result_path, tmp_file))
                if len(tmp) == 0:
                    continue
                class_data = np.array(tmp, ndmin=2)
                class_data = class_data[class_data[:, 0] == int(scene_frame) + 1, :]
                # frame_id, tt_id, x, y, z, dx, dy, dz, wx, wy, h
                class_data = class_data[:, [1, 2, 3, 4, 8, 9, 10]]
                ca_data.append(class_data)
            ca_data = np.row_stack(ca_data)
            files = json_data[scene][scene_frame]["files"]
            prj_mats = [np.array(cm) for cm in json_data[scene][scene_frame]["prj_mats"]]
            cam_exts = [np.array(cm) for cm in json_data[scene][scene_frame]["cam_exts"]]
            imgs = []
            for s, camstr in enumerate(sensors):
                img0 = cv2.imread(files[s])
                bbx_cam = bbxs[s][scene_frame][:, 0:5]
                bbx_cam = bbx_cam[bbx_cam[:, 4] > 0.5, 0:4]
                img = draw_on_images(prj_mats[s], cam_exts[s], ca_data, bbx_cam, img0, scene_frame, camstr)
                imgs.append(img)
            img_full_top_row = np.column_stack((imgs[4], imgs[1], imgs[5]))  # fleft, front, fright
            img_full_bottom_row = np.column_stack((imgs[2], imgs[0], imgs[3]))  # bleft, back, bright
            img_full = np.row_stack((img_full_top_row, img_full_bottom_row))
            dim = (int(img_full.shape[1] * scale_percent), int(img_full.shape[0] * scale_percent))
            resized = cv2.resize(img_full, dim, interpolation=cv2.INTER_AREA)  # resize image
            cv2.imshow('Image', resized)
            cv2.waitKey(0)
            out.write(resized)
        out.release()
    fd.close()


def format_sample_result(data_file="scenes_dict_val.json", result_path="../results/nuscenes"):
    # FairMOT-X, https://github.com/supperted825/FairMOT-X, https://doc.bdd100k.com/format.html
    # 1: pedestrian,  2: rider,  3: car,  4: truck,  5: bus,  6: train,  7: motorcycle, 8: bicycle
    # 9: traffic light,  10: traffic sign

    results = {}
    class_names = {1: "pedestrian", 2: "pedestrian", 3: "car", 4: "truck", 5: "bus", 6: "trailer", 7: "motorcycle",
                   8: "bicycle"}
    with open(data_file) as fd:
        json_data = json.load(fd)
        scenes = list(json_data.keys())
        for idx, scene in enumerate(scenes):
            print(idx, scene)
            for scene_frame in json_data[scene].keys():
                sample_token = json_data[scene][scene_frame]["sample_token"]
                sample_results = []
                for class_id in range(1, 9):
                    tmp_file = 'EST_{}_WORLD_CENTROID.txt'.format(scene + "_" + str(class_id))
                    tmp = np.loadtxt(os.path.join(result_path, tmp_file))
                    if len(tmp) == 0:
                        continue
                    class_data = np.array(tmp, ndmin=2)
                    class_data = class_data[class_data[:, 0] == int(scene_frame) + 1, :]
                    for idxc, (frame_id, tt_id, x, y, z, dx, dy, dz, wx, wy, h) in enumerate(class_data):
                        sample_result = {
                            "sample_token": sample_token,
                            "translation": [x, y, z],
                            "size": [wx, wy, h],
                            "rotation": [0.1886425013, -0.018708484, -0.0054355733, 0.981852562],
                            "velocity": [dx, dy],
                            "tracking_id": tt_id,
                            "tracking_name": class_names[class_id],
                            "tracking_score": 0.8
                        }
                        sample_results.append(sample_result)
                results[sample_token] = sample_results

    json_results = {
        "meta": {
            "use_camera": True,  # <bool>  -- Whether this submission uses camera data as an input.
            "use_lidar": False,  # <bool>  -- Whether this submission uses lidar data as an input.
            "use_radar": False,  # <bool>  -- Whether this submission uses radar data as an input.
            "use_map": False,  # <bool>  -- Whether this submission uses map data as an input.
            "use_external": False,  # <bool>  -- Whether this submission uses external data as an input.
        },
        "results": results
    }

    with open("./reid_ms_val.json", 'w') as fp:
        json.dump(json_results, fp)


def eval_msglmb_reid():
    format_sample_result()

    # Evaluation, Copy sample example from nuscenes/eval/tracking/evaluate.py, pip install motmetrics==1.1.3
    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes tracking results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--result_path', default="reid_ms_val.json", type=str, help='The submission as a JSON file.')
    parser.add_argument('--output_dir', type=str, default='./',
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str, default='D:/temp/nuScenes/v1.0-trainval_meta',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to the configuration file.'
                             'If no path given, the NIPS 2019 configuration will be used.')
    parser.add_argument('--render_curves', type=int, default=1,
                        help='Whether to render statistic curves to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    parser.add_argument('--render_classes', type=str, default='', nargs='+',
                        help='For which classes we render tracking results to disk.')
    args = parser.parse_args()

    result_path_ = os.path.expanduser(args.result_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    config_path = args.config_path
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)
    render_classes_ = args.render_classes

    if config_path == '':
        cfg_ = config_factory('tracking_nips_2019')
    else:
        with open(config_path, 'r') as _f:
            cfg_ = TrackingConfig.deserialize(json.load(_f))

    nusc_eval = TrackingEval(config=cfg_, result_path=result_path_, eval_set=eval_set_, output_dir=output_dir_,
                             nusc_version=version_, nusc_dataroot=dataroot_, verbose=verbose_,
                             render_classes=render_classes_)
    nusc_eval.main(render_curves=render_curves_)


if __name__ == '__main__':
    # (0) Download dataset from here: https://www.nuscenes.org/nuscenes#data-collection
    # (1) cd PATH_TO_WHERE_NUSCENES_BLOBS_WERE_DOWNLOADED_TO
    # (2) for f in *.tgz; do tar -xvf "$f"; done
    # nusc = NuScenes(version='v1.0-mini', dataroot='D:/temp/nuScenes/v1.0-mini', verbose=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot='/home/ubuntu/Desktop/nuScenes/v1.0-trainval_meta', verbose=True)
    get_scene_data2json(nusc)
    get_lidar_dets(nusc)
    gen_motchallenge_gt(nusc)

    # eval_msglmb_reid()
    # visual_tracking_results()
