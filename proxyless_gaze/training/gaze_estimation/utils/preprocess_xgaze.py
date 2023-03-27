import numpy as np
import h5py
from scipy.misc import face
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import random
from typing import List
import cv2
import pandas as pd
from tqdm import tqdm
import math
import pickle
from pprint import pprint
import platform

EYE_PATCH_SIZE = (60, 60)
FACE_PATCH_SIZE = (120, 120)
SYSTEM = platform.system()

if SYSTEM == "Windows":
    RAW_DATASET_PATH = "../../../eth_xgaze/"
    DATASET_STORE_PATH = "../../../xgaze_dataset"
else:
    RAW_DATASET_PATH = "/data/junyanli/datasets/xgaze_224"
    DATASET_STORE_PATH = "/dev/shm/xgaze_dataset"
os.makedirs(DATASET_STORE_PATH, exist_ok=True)

def get_dataloader(data_dir,
                   batch_size,
                   num_workers=8,
                   is_shuffle=True):
    # load dataset
    refer_list_file = os.path.join(data_dir, "data", "train_test_split.json")
    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)
    pprint(datastore["train"])
    # there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
    # train set: the training set includes 80 participants data
    # test set: the test set for cross-dataset and within-dataset evaluations
    # test_person_specific: evaluation subset for the person specific setting
    sub_folder_use = 'train'
    train_set = GazeDataset(dataset_path=data_dir, keys_to_use=datastore[sub_folder_use], sub_folder=sub_folder_use,
                            transform=None, is_shuffle=is_shuffle, is_load_label=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, drop_last=False)
    # train_loader = train_set
    return train_loader

class GazeDataset(Dataset):
    def __init__(self, dataset_path: str, keys_to_use: List[str] = None, sub_folder='', transform=None, is_shuffle=True,
                 index_file=None, is_load_label=True, augmentation=False):
        self.path = os.path.join(dataset_path, "data")
        self.hdfs = {}
        self.sub_folder = sub_folder
        self.is_load_label = is_load_label
        self.augmentation = augmentation

        # assert len(set(keys_to_use) - set(all_keys)) == 0
        # Select keys
        # TODO: select only people with sufficient entries?
        self.selected_keys = [k for k in keys_to_use]
        assert len(self.selected_keys) > 0

        for num_i in range(0, len(self.selected_keys)):
            file_path = os.path.join(self.path, self.sub_folder, self.selected_keys[num_i])
            self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
            # print('read file: ', os.path.join(self.path, self.selected_keys[num_i]))
            assert self.hdfs[num_i].swmr_mode

        # Construct mapping from full-data index to key and person-specific index
        if index_file is None:
            self.idx_to_kv = []
            for num_i in range(0, len(self.selected_keys)):
                n = self.hdfs[num_i]["face_patch"].shape[0]
                self.idx_to_kv += [(num_i, i) for i in range(n)]
        else:
            print('load the file: ', index_file)
            self.idx_to_kv = np.loadtxt(index_file, dtype=np.int)

        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

        if is_shuffle:
            random.shuffle(self.idx_to_kv)  # random the order to stable the training

        self.hdf = None
        self.transform = transform

        ## Devu Add
        self.__load_annotation(annotation_dir=dataset_path)
        self.hashtable = {}

    def __load_annotation(self, annotation_dir):
        ################## Parameters #################################################
        resize_factor = 8
        is_distor = False  # distortion is disable since it cost too much time, and the face is always in the center of image
        report_interval = 60
        is_over_write = True
        face_patch_size = 224
        ###########################################################################

        # load camera matrix
        self.camera_matrix = []
        self.camera_distortion = []
        self.cam_translation = []
        self.cam_rotation = []

        self.annotation_dir = annotation_dir
        print('Load the camera parameters')
        for cam_id in range(0, 18):
            file_name = f'{self.annotation_dir}/calibration/cam_calibration/' + 'cam' + str(cam_id).zfill(2) + '.xml'
            fs = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
            self.camera_matrix.append(fs.getNode('Camera_Matrix').mat())
            self.camera_distortion.append(fs.getNode('Distortion_Coefficients').mat()) # here we disable distortion
            self.cam_translation.append(fs.getNode('cam_translation').mat())
            self.cam_rotation.append(fs.getNode('cam_rotation').mat())
            fs.release()

        # load face model
        face_model_load = np.loadtxt(f'{self.annotation_dir}/calibration/face_model.txt')
        landmark_use = [20, 23, 26, 29, 15, 19]
        self.face_model = face_model_load[landmark_use, :]

    def __normalizeData_face(self, face_model, landmarks, hr, ht, gc, cam):
        ## normalized camera parameters
        focal_norm = 960  # focal length of normalized camera
        distance_norm = 300  # normalized distance between eye and camera
        roiSize = (448, 448)  # size of cropped eye image
        ## compute estimated 3D positions of the landmarks
        ht = ht.reshape((3, 1))
        gc = gc.reshape((3, 1))
        hR = cv2.Rodrigues(hr)[0]  # rotation matrix
        Fc = np.dot(hR, face_model.T) + ht
        two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
        mouth_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
        face_center = np.mean(np.concatenate((two_eye_center, mouth_center), axis=1), axis=1).reshape((3, 1))

        ## ---------- normalize image ----------
        distance = np.linalg.norm(face_center)  # actual distance between eye and original camera

        z_scale = distance_norm / distance
        cam_norm = np.array([
            [focal_norm, 0, roiSize[0] / 2],
            [0, focal_norm, roiSize[1] / 2],
            [0, 0, 1.0],
        ])
        S = np.array([  # scaling matrix
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, z_scale],
        ])

        hRx = hR[:, 0]
        forward = (face_center / distance).reshape(3)
        down = np.cross(forward, hRx)
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        R = np.c_[right, down, forward].T  # rotation matrix R

        W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))  # transformation matrix


        ## ---------- normalize rotation ----------
        hR_norm = np.dot(R, hR)  # rotation matrix in normalized space
        hr_norm = cv2.Rodrigues(hR_norm)[0]  # convert rotation matrix to rotation vectors

        ## ---------- normalize gaze vector ----------
        gc_normalized = gc - face_center  # gaze vector
        gc_normalized = np.dot(R, gc_normalized)
        gc_normalized = gc_normalized / np.linalg.norm(gc_normalized)

        # warp the facial landmarks
        num_point, num_axis = landmarks.shape
        det_point = landmarks.reshape([num_point, 1, num_axis])
        det_point_warped = cv2.perspectiveTransform(det_point, W)
        det_point_warped = det_point_warped.reshape(num_point, num_axis)


        head = hr_norm.reshape(1, 3)
        M = cv2.Rodrigues(head)[0]
        Zv = M[:, 2]
        head_2d = np.array([math.asin(Zv[1]), math.atan2(Zv[0], Zv[2]), 0.0]) # Add roll==0

        return head_2d, gc_normalized, det_point_warped, R

    def __len__(self):
        return len(self.idx_to_kv)

    def __del__(self):
        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

    def __segment_eye(self, image, lmks, eye='left', ow=64, oh=64):
        if eye=='left':
            # Left eye
            x1, y1 = lmks[36]
            x2, y2 = lmks[39]
        else: # right eye
            x1, y1 = lmks[42]
            x2, y2 = lmks[45]
           

        eye_width = 1.5 * np.linalg.norm(x1-x2)
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

        # center image on middle of eye
        translate_mat = np.asmatrix(np.eye(3))
        translate_mat[:2, 2] = [[-cx], [-cy]]
        
        # Scale
        scale = ow / (eye_width+1e-6)
        scale_mat = np.asmatrix(np.eye(3))
        scale_mat[0, 0] = scale_mat[1, 1] = scale
        
        # center image
        center_mat = np.asmatrix(np.eye(3))
        center_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]

        # Get rotated and scaled, and segmented image
        transform_mat =  center_mat * scale_mat * translate_mat
        eye_image = cv2.warpAffine(image, transform_mat[:2, :], (oh, ow))
    
        return eye_image

    def __crop_eye(self, face, fid, key, idx):
        cam_id = fid["cam_index"][idx][0]-1
        key_tmp = key.replace(".h5", "")
        df = pd.read_csv(f"{self.annotation_dir}/data/annotation_{self.sub_folder}/{key_tmp}.csv", header=None)
        # print("sssssssssssssssssSSS:", key)
        line = df.loc[idx,:].to_numpy()
        lmks = line[13:149].reshape(-1, 2).astype(float)
        gaze_label_3d = np.array([float(line[4]), float(line[5]), float(line[6])]).reshape(3, 1)  # gaze point on the screen coordinate system
        hr = np.array([float(line[7]), float(line[8]), float(line[9])]).reshape(3, 1)
        ht = np.array([float(line[10]), float(line[11]), float(line[12])]).reshape(3, 1)
        head_2D, gaze_norm, landmark_norm, mat_norm_face = \
                    self.__normalizeData_face(self.face_model, lmks, hr, ht, gaze_label_3d, self.camera_matrix[cam_id])
    
        landmark_norm = landmark_norm / 2  ## 448 size face ---> 224 size face

        left_eye = self.__segment_eye(face, landmark_norm, eye='left', ow=EYE_PATCH_SIZE[1], oh=EYE_PATCH_SIZE[0])
        right_eye = self.__segment_eye(face, landmark_norm, eye='right', ow=EYE_PATCH_SIZE[1], oh=EYE_PATCH_SIZE[0])

        return left_eye, right_eye, head_2D


    def __preprocess(self, eye):
        eye = cv2.resize(eye, EYE_PATCH_SIZE)
        eye = eye.astype(np.float32)
        eye *= 2.0 / 255.0
        eye -= 1.0
        # tranform to tensor order chw
        # eye = np.transpose(eye, (2, 0, 1))
        return eye

    def __getitem__(self, idx):
        key, idx = self.idx_to_kv[idx]
        if self.selected_keys[key] not in self.hashtable.keys():
            self.hdf = h5py.File(os.path.join(self.path, self.sub_folder, self.selected_keys[key]), 'r', swmr=True)
            self.hashtable[self.selected_keys[key]] = self.hdf
        else:
            self.hdf = self.hashtable[self.selected_keys[key]]

        assert self.hdf.swmr_mode

        # Get face image
        image = self.hdf['face_patch'][idx, :]
        left_eye, right_eye, head_2D = self.__crop_eye(image, self.hdf, self.selected_keys[key], idx)
        left_eye = self.__preprocess(left_eye)
        right_eye = self.__preprocess(right_eye)

        image = cv2.resize(image, FACE_PATCH_SIZE)
        image = image.astype(np.float32)
        image *= 2.0 / 255.0
        image -= 1.0

        # Get labels
        if self.is_load_label:
            gaze_label = self.hdf['face_gaze'][idx, :]
            gaze_label = gaze_label.astype('float')
            headpose_label = self.hdf['face_head_pose'][idx, :]
            gaze_label = np.rad2deg(gaze_label)
            headpose_label = np.rad2deg(headpose_label)
            return torch.FloatTensor(left_eye), torch.FloatTensor(right_eye), torch.FloatTensor(image), torch.FloatTensor(headpose_label), torch.FloatTensor(gaze_label), key, idx
        else:
            raise NotImplementedError


if __name__ == "__main__":
    if SYSTEM == "Windows":
        import sys
        sys.path.append("../")
        from visualize import draw_gaze
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
        train_loader = get_dataloader(RAW_DATASET_PATH, 4, 1, False)
    else:
        train_loader = get_dataloader(RAW_DATASET_PATH, 16, 16, False)

    for databatch in tqdm(train_loader):
        left, right, face, headpose, gaze, person_id, image_id = databatch
        left = left.numpy()
        right = right.numpy()
        face = face.numpy()
        headpose = headpose.numpy()
        gaze = gaze.numpy()
        person_id = person_id.numpy()
        image_id = image_id.numpy()

        left = ((left + 1)*128).astype(np.uint8)
        right = ((right + 1)*128).astype(np.uint8)
        face = ((face + 1)*128).astype(np.uint8)

        gray_left = []
        gray_right = []
        gray_face = []
        for raw, post in zip([left, right, face], [gray_left, gray_right, gray_face]):
            for l in raw:
                l = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
                l = cv2.equalizeHist(l)
                post.append(l)
        left = np.array(gray_left)
        right = np.array(gray_right)
        face = np.array(gray_face)
        
        for l, r, f, h, g, pid, iid in zip(left, right, face, headpose, gaze, person_id, image_id):
            record = [l, r, f, g]
            filename = f"{str(pid).zfill(4)}_{str(iid).zfill(8)}_{int(g[0])}_{int(g[1])}_{int(h[0])}_{int(h[1])}.pkl"
            with open(os.path.join(DATASET_STORE_PATH, filename), "wb") as f:
                pickle.dump(record, f)
        
        if SYSTEM == "Windows":
            tqdm.write(f"{left.shape} {right.shape} {face.shape}")
            left = left[2]
            right = right[2]
            face = face[2]
            headpose = headpose[2]
            gaze = gaze[2]
            
            gaze = np.deg2rad(gaze)
            left = draw_gaze(left, (30, 30), gaze, length=15)
            right = draw_gaze(right, (30, 30), gaze, length=15)
            face = draw_gaze(face, (60, 60), gaze, length=30)
            concat = np.hstack([left, right])

            cv2.imshow("Image", concat)
            cv2.imshow("Face", face)

            k = cv2.waitKey(0)
            if k == 27:
                cv2.destroyAllWindows()
                break
        