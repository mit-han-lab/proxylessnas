import pickle
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import os.path as osp
import numpy as np
from utils.normalize_data import normalizeData
import cv2
import scipy.io as sio
import glob

class XGazeDataset(data.Dataset):
    def __init__(self, datapath, is_train):
        self.file_list = glob.glob(osp.join(datapath, "*.pkl"))
        person_ids = set()
        for name in self.file_list:
            name = osp.basename(name)
            person_id = name.split("_")[0]
            person_ids.add(person_id)
        person_ids = list(person_ids)
        person_ids.sort()
        if is_train:
            person_ids = person_ids[:int(0.9*len(person_ids))]
        else:
            person_ids = person_ids[int(0.9*len(person_ids)):]
        file_list = []
        for i, name in enumerate(self.file_list):
            name = osp.basename(name)
            person_id = name.split("_")[0]
            if person_id in person_ids:
                file_list.append(self.file_list[i])
        self.file_list = file_list
        self.dataset_len = len(self.file_list)
        self.to_tensor = transforms.ToTensor()
        self.person_ids = person_ids
    
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, index):
        with open(self.file_list[index], "rb") as f:
            leye, reye, face, label = pickle.load(f)
        leye = self.to_tensor(leye)
        reye = self.to_tensor(reye)
        face = self.to_tensor(face)
        label = torch.FloatTensor(label)
        return [leye, reye, face, label]
    

class MPIIGazeDataset(data.Dataset):
    def __init__(self, anno_filename, is_train=False):
        with open(anno_filename, "r") as f:
            self.samples = f.readlines()
        self.dataset_len = len(self.samples)
        self.is_train = is_train
        self.to_tensor = transforms.ToTensor()
    
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, index):
        sample = self.samples[index].strip().split()
        leye = Image.open(sample[0])
        reye = Image.open(sample[1])
        face = Image.open(sample[2])
        label = torch.tensor(list(map(float, sample[3:]))) * 100.0
        
        if self.is_train:
            p = np.random.randint(2)
            if p == 1:
                face = face.transpose(Image.FLIP_LEFT_RIGHT)
                leye = leye.transpose(Image.FLIP_LEFT_RIGHT)
                reye = reye.transpose(Image.FLIP_LEFT_RIGHT)
                leye, reye = reye, leye
                label[-1] = -label[-1]
        leye = self.to_tensor(leye)
        reye = self.to_tensor(reye)
        face = self.to_tensor(face)
        
        return [leye, reye, face, label]

class AdvancedMPIIGazeDataset(data.Dataset):
    def __init__(self, dataset_dir, annotation_filename, is_train=False, enable_flip=True, enable_jitter=True):
        with open(annotation_filename, "r") as f:
            _samples = f.readlines()
            self.samples = [sample.strip().split() for sample in _samples]
        self.camera_matrix = {}
        self.camera_distortion = {}
        for subject_id in range(0, 15):
            subject_id = f"p{str(subject_id).zfill(2)}"
            cameraCalib = sio.loadmat(osp.join(dataset_dir, subject_id, "Calibration", "Camera.mat"))
            camera_matrix = cameraCalib['cameraMatrix']
            camera_distortion = cameraCalib['distCoeffs']
            self.camera_matrix[subject_id] = camera_matrix
            self.camera_distortion[subject_id] = camera_distortion
        self.face_model = sio.loadmat(osp.join(dataset_dir, '6 points-based face model.mat'))['model']
        self.dataset_len = len(self.samples)
        self.is_train = is_train
        self.to_tensor = transforms.ToTensor()
        self.enable_flip = enable_flip
        self.enable_jitter = enable_jitter

    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, index):
        anno = self.samples[index]
        filepath = anno[0]
        subject_id = anno[-1]
        
        landmarks = np.array(list(map(int, anno[3:15]))).reshape(6, 2)
        face_center_3d = np.array(list(map(float, anno[21:24])))
        gc = np.array(list(map(float, anno[24:27])))
        
        img_original = cv2.imread(filepath)
        img = cv2.undistort(img_original, self.camera_matrix[subject_id], self.camera_distortion[subject_id])
        
        num_pts = self.face_model.shape[1]
        landmarks = landmarks.astype(np.float32)
        landmarks = landmarks.reshape(num_pts, 1, 2)
        hr = np.array(list(map(float, anno[15:18]))).reshape(3,1)
        ht = np.array(list(map(float, anno[18:21]))).reshape(3,1)
        data = normalizeData(img, self.face_model, hr, ht, gc, self.camera_matrix[subject_id], face_center_3d, jitter=(self.is_train & self.enable_jitter))
        leye_image, reye_image, face_image, gaze_xyz = data
        x, y, z = list(gaze_xyz.reshape(3))
        gaze_theta = np.arcsin(-y)
        gaze_phi = np.arctan2(-x, -z)
        
        if self.is_train and self.enable_flip:
            p = np.random.randint(2)
            if p == 1:
                face_image = cv2.flip(face_image, 1)
                leye_image = cv2.flip(leye_image, 1)
                reye_image = cv2.flip(reye_image, 1)
                leye_image, reye_image = reye_image, leye_image
                gaze_phi = -gaze_phi
        
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10,3))
        # plt.subplot(1,3,1)
        # plt.imshow(face_image, cmap="gray")
        # plt.subplot(1,3,2)
        # plt.imshow(leye_image, cmap="gray")
        # plt.subplot(1,3,3)
        # plt.imshow(reye_image, cmap="gray")
        # print(f"{np.rad2deg(gaze_theta):.2f} {np.rad2deg(gaze_phi):.2f}")
        # plt.tight_layout()
        # plt.show()
        label = torch.tensor([float(gaze_theta), float(gaze_phi)]) * 100.0
        leye = self.to_tensor(leye_image)
        reye = self.to_tensor(reye_image)
        face = self.to_tensor(face_image)
        return [leye, reye, face, label]

if __name__ == "__main__":
    cnt = 0
    from visualize import draw_gaze
    train_dataset = XGazeDataset("/dev/shm/xgaze_dataset", is_train=True)
    test_dataset = XGazeDataset("/dev/shm/xgaze_dataset", is_train=False)
    train_dataloader = data.DataLoader(train_dataset, 16, shuffle=True)
    for data in train_dataloader:
        cnt += 1
        if cnt == 51:
            break
        left, right, face, gaze = data
        print(data[0].shape, data[1].shape, data[2].shape, data[3].shape)
        left = np.transpose(left[0].numpy(), (1,2,0))
        right = np.transpose(right[0].numpy(), (1,2,0))
        face = np.transpose(face[0].numpy(), (1,2,0))
        gaze = gaze[0].numpy()

        left = (left*256).astype(np.uint8)
        right = (right*256).astype(np.uint8)
        face = (face*256).astype(np.uint8)

        gaze = np.deg2rad(gaze)
        # left = draw_gaze(left, (30, 30), gaze, length=15)
        # right = draw_gaze(right, (30, 30), gaze, length=15)
        # face = draw_gaze(face, (60, 60), gaze, length=30)
        concat = np.hstack([left, right])
        gaze = np.rad2deg(gaze)
        print(left.shape, right.shape, face.shape, gaze)
        cv2.imwrite(f"samples/{gaze[0]}_{gaze[1]}_eye.png", concat)
        cv2.imwrite(f"samples/{gaze[0]}_{gaze[1]}_face.png", face)
