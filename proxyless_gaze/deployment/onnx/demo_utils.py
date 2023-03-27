#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os
import numpy as np
import platform
import time
import ctypes
import cv2

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
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets

def demo_postprocess(outputs, img_size, p6=False):

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
        if "Windows" in platform.platform():
            freq = ctypes.c_longlong(0)
            ctypes.windll.kernel32.QueryPerformanceFrequency(ctypes.byref(freq))
            self.freq = freq.value
            self.get_current_timestamp = self.get_current_timestamp_windows
        else:
            self.get_current_timestamp = self.get_current_timestamp_linux
    
    def get_current_timestamp_windows(self):
        freq = ctypes.c_longlong(0)
        ctypes.windll.kernel32.QueryPerformanceCounter(ctypes.byref(freq))
        return freq.value / self.freq

    def get_current_timestamp_linux(self):
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
        color = (0,0,0)
        cv2.putText(img, f"Size: {img.shape[1]}x{img.shape[0]}", (10,35), font, 0.8, color, 2)
        cv2.putText(img, f"FPS: {int(1/(self.get_record_s(['whole_pipeline'])+1e-5))}", (10,60), font, 0.8, color, 2)
        cv2.putText(img, f"Total: {self.get_record_ms(['whole_pipeline'])}ms", (10,85), font, 0.8, color, 2)
        
        cnt = 0
        if "face_detection" in self.record:
            cnt += 1
            cv2.putText(img, f"Detection: {self.get_record_ms(['face_detection'])}ms", 
                        (10,85+25*cnt), font, 0.6, color, 2)
        if "face_detection_postprocess" in self.record:
            cnt += 1
            cv2.putText(img, f"Detection NMS: {self.get_record_ms(['face_detection_postprocess'])}ms", 
                        (10,85+25*cnt), font, 0.6, color, 2)
        
        if "landmark_detection" in self.record:
            cnt += 1
            cv2.putText(img, f"Landmark: {self.get_record_ms(['landmark_detection'])}ms", 
                        (10,85+25*cnt), font, 0.6, color, 2)
        if "gaze_estimation" in self.record:
            cnt += 1
            cv2.putText(img, f"Gaze: {self.get_record_ms(['gaze_estimation'])}ms", 
                        (10,85+25*cnt), font, 0.6, color, 2)
        if "gaze_estimation_preprocess" in self.record:
            cnt += 1
            cv2.putText(img, f"Gaze Preprocess: {self.get_record_ms(['gaze_estimation_preprocess'])}ms", 
                        (10,85+25*cnt), font, 0.6, color, 2)
        if "visualize" in self.record:
            cnt += 1
            cv2.putText(img, f"Visualize: {self.get_record_ms(['visualize'])}ms", 
                        (10,85+25*cnt), font, 0.6, color, 2)
        
        return img
      
def draw_gaze(image_in, eye_pos, pitchyaw, length=15.0, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                   tuple(np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)

    return image_out

