#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import numpy as np
import math
import time
import cv2
import scipy.linalg as sla
from tqdm import trange, tqdm

face_model = np.float32([
    [-63.833572,  63.223045,  41.1674],  # RIGHT_EYEBROW_RIGHT,
    [-12.44103,  66.60398,  64.561584],  # RIGHT_EYEBROW_LEFT,
    [12.44103,  66.60398,  64.561584],  # LEFT_EYEBROW_RIGHT,
    [63.833572,  63.223045,  41.1674],  # LEFT_EYEBROW_LEFT,
    [-49.670784,  51.29701,  37.291245],  # RIGHT_EYE_RIGHT,
    [-16.738844,  50.439426,  41.27281],  # RIGHT_EYE_LEFT,
    [16.738844,  50.439426,  41.27281],  # LEFT_EYE_RIGHT,
    [49.670784,  51.29701,  37.291245],  # LEFT_EYE_LEFT,
    [-18.755981,  13.184412,  57.659172],  # NOSE_RIGHT,
    [18.755981,  13.184412,  57.659172],  # NOSE_LEFT,
    [-25.941687, -19.458733,  47.212223],  # MOUTH_RIGHT,
    [25.941687, -19.458733,  47.212223],  # MOUTH_LEFT,
    [0., -29.143637,  57.023403],  # LOWER_LIP,
    [0., -69.34913,  38.065376]  # CHIN
])

cam_w, cam_h = 640, 480
c_x = cam_w / 2
c_y = cam_h / 2
f_x = c_x / np.tan(60 / 2 * np.pi / 180)
f_y = f_x
camera_matrix = np.float32([[f_x, 0.0, c_x], [0.0, f_y, c_y], [0.0, 0.0, 1.0]])
camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])
# normalized camera parameters
focal_norm = 960  # focal length of normalized camera
distance_norm_eye = 700  # normalized distance between eye and camera
distance_norm_face = 1200  # normalized distance between face and camera
roiSize_eye = (60, 60)  # size of cropped eye image
roiSize_face = (120, 120)  # size of cropped face image
camera_matrix_inv = sla.inv(camera_matrix)


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=True):
    """Multiclass NMS implemented in Numpy"""
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware
    return nms_method(boxes, scores, nms_thr, score_thr)


def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None],
                valid_cls_inds[keep, None]], 1
        )
    return dets


def yolox_preprocess(img, input_size=(160, 128), swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones(
            (input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    padded_img = padded_img.transpose(swap).astype(np.float32)
    return padded_img, r


def demo_postprocess(outputs, img_size=(160, 128), p6=False):

    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


def normalizeDataForInference(img, hr, ht):

    # compute estimated 3D positions of the landmarks
    ht = ht.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]  # rotation matrix
    Fc = np.dot(hR, face_model.T) + ht  # 3D positions of facial landmarks

    # center of eye on the left side of image (right eye)
    re = 0.5*(Fc[:, 4] + Fc[:, 5]).T
    # center of eye on the right side of image (left eye)
    le = 0.5*(Fc[:, 6] + Fc[:, 7]).T
    fe = (1./6.)*(Fc[:, 4] + Fc[:, 5] + Fc[:, 6] +
                  Fc[:, 7] + Fc[:, 10] + Fc[:, 11]).T

    # normalize each eye
    data = []
    face_R = None
    for distance_norm, roiSize, et in zip([distance_norm_eye, distance_norm_eye, distance_norm_face], [roiSize_eye, roiSize_eye, roiSize_face], [re, le, fe]):
        # ---------- normalize image ----------
        # actual distance between eye and original camera
        distance = np.linalg.norm(et)

        z_scale = distance_norm/distance
        cam_norm = np.array([
            [focal_norm, 0, roiSize[0]/2],
            [0, focal_norm, roiSize[1]/2],
            [0, 0, 1.0],
        ])
        S = np.array([  # scaling matrix
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, z_scale],
        ])

        hRx = hR[:, 0]
        forward = (et/distance).reshape(3)
        down = np.cross(forward, hRx)
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        R = np.c_[right, down, forward].T  # rotation matrix R
        W = np.dot(np.dot(cam_norm, S), np.dot(
            R, camera_matrix_inv))  # transformation matrix
        # use cv2.INTER_NEAREST is much fatser
        img_warped = cv2.warpPerspective(
            img, W, roiSize, flags=cv2.INTER_NEAREST)  # image normalization
        data.append(img_warped)

        if distance_norm == distance_norm_face:
            face_R = R
    return data, face_R


def extract_critical_landmarks(landmark, pt_num=14):
    TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
    critical_landmarks = landmark[TRACKED_POINTS]
    critical_landmarks = np.array(critical_landmarks)
    return critical_landmarks


def euler_to_vec(theta, phi):
    x = -1 * np.cos(theta) * np.sin(phi)
    y = -1 * np.sin(theta)
    z = -1 * np.cos(theta) * np.cos(phi)
    vec = np.array([x, y, z])
    vec = vec / np.linalg.norm(vec)
    return vec


def vec_to_euler(x, y, z):
    theta = np.arcsin(-y)
    phi = np.arctan2(-x, -z)
    return theta, phi


def rtvec_to_euler(rvec, tvec, unit="radian"):
    rvec_matrix = cv2.Rodrigues(rvec)[0]
    proj_matrix = np.hstack((rvec_matrix, tvec))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
    if unit == "degree":
        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))
    return pitch, yaw, roll


def estimateHeadPose(landmarks):
    landmarks = extract_critical_landmarks(landmarks)
    ret, rvec, tvec = cv2.solvePnP(
        face_model, landmarks, camera_matrix, camera_distortion)
    return rvec, tvec


class Timer():
    def __init__(self, saved_n=100):
        self.record = {}
        self.start_time = {}
        self.end_time = {}
        self.current_name = None
        self.saved_n = saved_n
        # self.weights = list(range(saved_n))
        # self.weights.reverse()
        # self.weights = np.exp(-np.array(self.weights))
        # if "Windows" in platform.platform():
        #     freq = ctypes.c_longlong(0)
        #     ctypes.windll.kernel32.QueryPerformanceFrequency(ctypes.byref(freq))
        #     self.freq = freq.value
        #     self.get_current_timestamp = self.get_current_timestamp_windows
        # else:
        #     self.get_current_timestamp = self.get_current_timestamp_linux

    # def get_current_timestamp_windows(self):
    #     freq = ctypes.c_longlong(0)
    #     ctypes.windll.kernel32.QueryPerformanceCounter(ctypes.byref(freq))
    #     return freq.value / self.freq

    # def get_current_timestamp_linux(self):
    #     return time.time()

    def get_current_timestamp(self):
        return time.time()

    def start_record(self, name=None):
        if name is None:
            name = "default"
        self.current_name = name
        self.start_time[name] = self.get_current_timestamp()

    def end_record(self, name=None):
        if name is None:
            name = self.current_name
        self.end_time[name] = self.get_current_timestamp()
        if name not in self.record:
            # self.record[name] = [0] * self.saved_n
            self.record[name] = []
        self.record[name].append(self.end_time[name] - self.start_time[name])
        if len(self.record[name]) > self.saved_n:
            self.record[name].pop(0)

    def get_record_s(self, name=None) -> float:
        if name is None:
            name = self.record.keys()
        return sum([np.mean(self.record[n]) for n in name])

    def get_record_ms(self, name=None) -> float:
        return round(self.get_record_s(name) * 1000.0, 2)

    def clear_all(self):
        self.start_time = {}
        self.end_time = {}
        self.record = {}

    def print_on_image(self, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 0, 0)
        cv2.putText(
            img, f"Size: {img.shape[1]}x{img.shape[0]}", (10, 35), font, 0.8, color, 2)
        cv2.putText(
            img, f"FPS: {int(1/(self.get_record_s(['whole_pipeline'])+1e-5))}", (10, 60), font, 0.8, color, 2)
        cv2.putText(
            img, f"Total: {self.get_record_ms(['whole_pipeline'])}ms", (10, 85), font, 0.8, color, 2)

        cnt = 0
        if "face_detection" in self.record:
            cnt += 1
            cv2.putText(img, f"Detection: {self.get_record_ms(['face_detection'])}ms",
                        (10, 85+25*cnt), font, 0.6, color, 2)
        if "face_detection_postprocess" in self.record:
            cnt += 1
            cv2.putText(img, f"Detection NMS: {self.get_record_ms(['face_detection_postprocess'])}ms",
                        (10, 85+25*cnt), font, 0.6, color, 2)

        if "landmark_detection" in self.record:
            cnt += 1
            cv2.putText(img, f"Landmark: {self.get_record_ms(['landmark_detection'])}ms",
                        (10, 85+25*cnt), font, 0.6, color, 2)
        if "gaze_estimation" in self.record:
            cnt += 1
            cv2.putText(img, f"Gaze: {self.get_record_ms(['gaze_estimation'])}ms",
                        (10, 85+25*cnt), font, 0.6, color, 2)
        if "gaze_estimation_preprocess" in self.record:
            cnt += 1
            cv2.putText(img, f"Gaze Preprocess: {self.get_record_ms(['gaze_estimation_preprocess'])}ms",
                        (10, 85+25*cnt), font, 0.6, color, 2)
        if "visualize" in self.record:
            cnt += 1
            cv2.putText(img, f"Visualize: {self.get_record_ms(['visualize'])}ms",
                        (10, 85+25*cnt), font, 0.6, color, 2)

        return img


def draw_gaze(image_in, eye_pos, pitchyaw, length=15.0, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                    tuple(
                        np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int)), color,
                    thickness, cv2.LINE_AA, tipLength=0.2)

    return image_out


def warmup(face_detector, face_landmark_detector, gaze_estimator, image_path="./warmup.png", warmup_num=400):
    tqdm.write("start warmup...")
    tqdm.write(f"warmup iteration: {warmup_num}")
    for _ in trange(warmup_num):
        frame = cv2.imread(image_path)
        faces = face_detector.inference(frame)
        face = faces[0]
        x1, y1, x2, y2 = face[:4]
        face = np.array([x1, y1, x2, y2, face[-1]])
        landmark = face_landmark_detector.inference(frame, face)
        gaze_pitchyaw, rvec, tvec = gaze_estimator.inference(frame, landmark)
    tqdm.write("warmup done...")
