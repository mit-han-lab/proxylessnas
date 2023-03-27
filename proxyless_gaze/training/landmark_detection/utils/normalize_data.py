# -*- coding: utf-8 -*-
"""
######################################################################################################################################
This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy of this license,
visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

Any publications arising from the use of this software, including but
not limited to academic journal and conference publications, technical
reports and manuals, must cite at least one of the following works:

Revisiting Data Normalization for Appearance-Based Gaze Estimation
Xucong Zhang, Yusuke Sugano, Andreas Bulling
in Proc. International Symposium on Eye Tracking Research and Applications (ETRA), 2018
######################################################################################################################################
"""

import os
import cv2
import numpy as np
import csv
import scipy.io as sio
import matplotlib.pyplot as plt

def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True, flags=cv2.SOLVEPNP_EPNP):
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=flags)
    ## further optimize
    if iterate:
        ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)

    return rvec, tvec

def normalizeData(img, face, hr, ht, gc, cam, face_center=None):
    ## normalized camera parameters
    focal_norm = 960 # focal length of normalized camera
    distance_norm_eye = 600 # normalized distance between eye and camera
    distance_norm_face = 1000 # normalized distance between face and camera
    roiSize_eye = (60, 36) # size of cropped eye image
    roiSize_face = (120, 120) # size of cropped face image

    img_u = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3,1))
    gc = gc.reshape((3,1))
    hR = cv2.Rodrigues(hr)[0] # rotation matrix
    Fc = np.dot(hR, face) + ht # 3D positions of facial landmarks

    re = 0.5*(Fc[:,0] + Fc[:,1]).reshape((3,1)) # center of eye on the left side of image (right eye)
    le = 0.5*(Fc[:,2] + Fc[:,3]).reshape((3,1)) # center of eye on the right side of image (left eye)
    if face_center is not None:
        fe = np.array(face_center).reshape((3,1))
    else:
        fe = Fc.mean(-1).reshape((3,1))
    
    ## normalize each eye
    data = []
    for distance_norm, roiSize, et in zip([distance_norm_eye, distance_norm_eye, distance_norm_face], [roiSize_eye, roiSize_eye, roiSize_face], [re, le, fe]):
        ## ---------- normalize image ----------
        distance = np.linalg.norm(et) # actual distance between eye and original camera
        
        z_scale = distance_norm/distance
        cam_norm = np.array([
            [focal_norm, 0, roiSize[0]/2],
            [0, focal_norm, roiSize[1]/2],
            [0, 0, 1.0],
        ])
        S = np.array([ # scaling matrix
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, z_scale],
        ])
        
        hRx = hR[:,0]
        forward = (et/distance).reshape(3)
        down = np.cross(forward, hRx)
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        R = np.c_[right, down, forward].T # rotation matrix R
        
        W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam))) # transformation matrix
        
        img_warped = cv2.warpPerspective(img_u, W, roiSize) # image normalization
        img_warped = cv2.equalizeHist(img_warped)
        
        # ## ---------- normalize rotation ----------
        # hR_norm = np.dot(R, hR) # rotation matrix in normalized space
        # hr_norm = cv2.Rodrigues(hR_norm)[0] # convert rotation matrix to rotation vectors
        
        # ## ---------- normalize gaze vector ----------
        # gc_normalized = gc - et # gaze vector
        # # For modified data normalization, scaling is not applied to gaze direction, so here is only R applied.
        # # For original data normalization, here should be:
        # # "M = np.dot(S,R)
        # # gc_normalized = np.dot(R, gc_normalized)"
        # gc_normalized = np.dot(R, gc_normalized)
        # gc_normalized = gc_normalized/np.linalg.norm(gc_normalized)

        data.append(img_warped)
        
        ## compute gaze direction in normalized space
        if distance_norm == distance_norm_face: # only compute when it is time for face
            et = Fc[:,:4].mean(-1).reshape((3,1)) # virtual start point of gaze
            gc_normalized = gc - et # 3d gaze vector in camera coordinate
            gc_normalized = np.dot(R, gc_normalized)
            gc_normalized = gc_normalized/np.linalg.norm(gc_normalized)
            data.append(gc_normalized)
    
    return data

if __name__ == '__main__':
    ## load calibration data, these paramters can be obtained by camera calibration functions in OpenCV
    cameraCalib = sio.loadmat('./data/calibration/cameraCalib.mat')
    camera_matrix = cameraCalib['cameraMatrix']
    camera_distortion = cameraCalib['distCoeffs']

    # load example
    filepath = os.path.join('./data/example/day01_0087.jpg')
    img_original = cv2.imread(filepath)
    img = cv2.undistort(img_original, camera_matrix, camera_distortion)
    # load the detected facial landmarks
    # this code does not contain facial landmark detection
    landmarks = np.array([[551, 408], [603, 405], [698, 398], [755, 393], [603, 566], [724, 557]])

    # estimate head pose
    # load the generic face model, which includes 6 facial landmarks: four eye corners and two mouth corners
    face = sio.loadmat('./data/faceModelGeneric.mat')['model']
    num_pts = face.shape[1]
    facePts = face.T.reshape(num_pts, 1, 3)
    landmarks = landmarks.astype(np.float32)
    landmarks = landmarks.reshape(num_pts, 1, 2)
    hr, ht = estimateHeadPose(landmarks, facePts, camera_matrix, camera_distortion)

    # load 3D gaze target position in camera coordinate system
    gc = np.array([-127.790719, 4.621111, -12.025310])  # 3D gaze taraget position

    # data normalization for left and right eye image
    data = normalizeData(img, face, hr, ht, gc, camera_matrix)
    
    x, y, z = data[-1]
    gaze_theta = np.arcsin(-y) # yaw
    gaze_phi = np.arctan2(-x, -z) # pitch
    
    print(len(data))
    print(data[-1])
    print(gaze_theta, gaze_phi)

    # # show results of right eye image
    # label = data[0][2]
    # print('The label is: ', label)
    # # convert label to euler angle
    # gaze_theta = np.arcsin((-1) * label[1])
    # gaze_phi = np.arctan2((-1) * label[0], (-1) * label[2])

    # show normalized image
    plt.subplot(2,2,1)
    plt.imshow(data[0], cmap=plt.cm.gray)
    plt.subplot(2,2,2)
    plt.imshow(data[1], cmap=plt.cm.gray)
    plt.subplot(2,1,2)
    plt.imshow(data[2], cmap=plt.cm.gray)
    plt.tight_layout()
    plt.show()
    
    cv2.imwrite("leye.bmp", data[0])
    cv2.imwrite("reye.bmp", data[1])
    cv2.imwrite("face.bmp", data[2])
    