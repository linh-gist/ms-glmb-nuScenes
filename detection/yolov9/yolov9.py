from ultralytics import YOLO
import cv2
import numpy as np
import json
import os
import os.path as osp


def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)


def detect_save():
    # Build a YOLOv9c model from scratch
    # model = YOLO('yolov9c.yaml')
    # Build a YOLOv9c model from pretrained weight
    model = YOLO('yolov9e.pt')
    # Display model information (optional)
    model.info()
    # Train the model on the COCO8 example dataset for 100 epochs
    # results = model.train(data='coco8.yaml', epochs=100, imgsz=640)
    # Run inference with the YOLOv9c model on the 'bus.jpg' image
    # results = model('path/to/bus.jpg')

    dict1 = {}
    sensors = ["CAM_BACK", "CAM_FRONT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]
    json_file = "../scenes_dict_val.json"
    fd = open(json_file)
    json_data = json.load(fd)
    scenes = list(json_data.keys())
    for scene in scenes:
        scene_data = json_data[scene]
        frames = list(scene_data.keys())
        det_dicts = [dict() for s in sensors]
        for frame in frames:
            files = scene_data[frame]["files"]
            if len(files) != len(sensors):
                print("Error", len(files), len(sensors))
                exit(1)
            for cam, file in enumerate(files):
                img0 = cv2.imread(file)
                # results1 = model(img0)
                results = model.predict(img0, verbose=False)
                dets = results[0].boxes.data.detach().cpu().numpy()
                # Output detection [Left, Top, Right, Bottom, score, label]
                tracking_names = ["bicycle", "bus", "car", "motorcycle", "person", "trailer", "truck"]
                tracking_dets = []
                for l, t, r, b, s, label in dets:
                    if model.names[label] in tracking_names:
                        tracking_dets.append([l, t, r, b, s, label])
                if len(tracking_dets) == 0:
                    tracking_dets = np.empty((0, 6))
                else:
                    tracking_dets = np.array(tracking_dets)
                det_dicts[cam][str(frame)] = np.array(tracking_dets)

                img1 = np.copy(img0)
                # for l, t, r, b, conf, class_conf, cls_id in dets_feats[:, :7]:
                #     if cls_id not in dict1.keys() and conf > 0.8:
                #         print(cls_id)
                #         dict1[cls_id] = cls_id
                # l, t, r, b = int(l), int(t), int(r), int(b)
                # img1 = cv2.rectangle(img1, (l, t), (r, b), color=(255, 255, 255), thickness=2)
                # img1 = cv2.putText(img1, str(round(conf, 2)), org=(l, t), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #                    fontScale=0.65, color=(0, 255, 255), thickness=2)
                # cv2.imshow(scene, img1)
                # cv2.waitKey(0)

                if cam == 0:
                    for l, t, r, b, conf, label in np.copy(tracking_dets):
                        l, t, r, b = int(l), int(t), int(r), int(b)
                        img0 = cv2.rectangle(img0, (l, t), (r, b), color=(255, 255, 255), thickness=2)
                        img0 = cv2.putText(img0, str(round(conf, 2)), org=(l, t), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                           fontScale=0.65, color=(0, 255, 255), thickness=2)
                        img0 = cv2.putText(img0, model.names[label], org=(r, b), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                           fontScale=0.65, color=(0, 255, 255), thickness=2)
                    cv2.imshow(scene, img0)
                    cv2.waitKey(1)
        cv2.destroyAllWindows()
        for idx, det_dict in enumerate(det_dicts):
            save_dir = os.path.join("./nuscenes", scene)
            mkdir_if_missing(save_dir)
            np.savez(os.path.join(save_dir, "Cam_" + str(idx + 1) + ".npz"), **det_dict)  # data is a dict here
    print(json_data)
    fd.close()


if __name__ == '__main__':
    detect_save()
