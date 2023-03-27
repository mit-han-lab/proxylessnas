import os
import os.path as osp
import argparse
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import scipy.io as sio
import cv2
import numpy as np
from normalize_data import estimateHeadPose, normalizeData

parser = argparse.ArgumentParser(description='Preprocess MPIIGaze dataset')
parser.add_argument('-src-path', type=str, required=True, help='the path to original dataset')
parser.add_argument('-dest-path', type=str, required=True, help='the path to preprocessed dataset')
parser.add_argument('-nproc', type=int, default=1, help='multiprocess core number')
args = parser.parse_args()

subject_num = 15
subject_ids = list(range(subject_num))
face_model = sio.loadmat(osp.join(args.src_path, '6 points-based face model.mat'))['model']

def handle_subject(subject_id):
    subject_id = 'p'+str(subject_id).zfill(2)
    tqdm.write(f"Process subject {subject_id}...")
    subject_dir = osp.join(args.src_path, subject_id)
    os.makedirs(osp.join(args.dest_path, subject_id), exist_ok=True)

    cameraCalib = sio.loadmat(osp.join(subject_dir, "Calibration", "Camera.mat"))
    camera_matrix = cameraCalib['cameraMatrix']
    camera_distortion = cameraCalib['distCoeffs']
    
    converted_gts = []
    with open(osp.join(subject_dir, subject_id+'.txt'), "r") as f:
        annotations = f.readlines()
        for anno in tqdm(annotations):
            anno = anno.strip().split()
            filepath = osp.join(subject_dir, anno[0])
            landmarks = np.array(list(map(int, anno[3:15]))).reshape(6, 2)
            face_center_3d = np.array(list(map(float, anno[21:24])))
            gc = np.array(list(map(float, anno[24:27])))
            
            img_original = cv2.imread(filepath)
            img = cv2.undistort(img_original, camera_matrix, camera_distortion)
            
            num_pts = face_model.shape[1]
            facePts = face_model.T.reshape(num_pts, 1, 3)
            landmarks = landmarks.astype(np.float32)
            landmarks = landmarks.reshape(num_pts, 1, 2)
            hr, ht = estimateHeadPose(landmarks, facePts, camera_matrix, camera_distortion)
            
            data = normalizeData(img, face_model, hr, ht, gc, camera_matrix, face_center_3d)
            leye_image, reye_image, face_image, gaze_xyz = data
            x, y, z = list(gaze_xyz.reshape(3))
            gaze_theta = np.arcsin(-y)
            gaze_phi = np.arctan2(-x, -z)
            
            leye_full_path = osp.join(args.dest_path, subject_id, anno[0].replace('/', '_')[:-4]+"_leye.bmp")
            reye_full_path = osp.join(args.dest_path, subject_id, anno[0].replace('/', '_')[:-4]+"_reye.bmp")
            face_full_path = osp.join(args.dest_path, subject_id, anno[0].replace('/', '_')[:-4]+"_face.bmp")
            leye_full_path = osp.abspath(leye_full_path)
            reye_full_path = osp.abspath(reye_full_path)
            face_full_path = osp.abspath(face_full_path)
            
            cv2.imwrite(leye_full_path, leye_image)
            cv2.imwrite(reye_full_path, reye_image)
            cv2.imwrite(face_full_path, face_image)
            converted_gts.append(" ".join([leye_full_path, reye_full_path, face_full_path, str(gaze_theta), str(gaze_phi)]))
    return converted_gts

if args.nproc == 1:
    all_converted_gts = []
    for subject_id in subject_ids:
        all_converted_gts.append(handle_subject(subject_id))
else:
    all_converted_gts = process_map(handle_subject, subject_ids, max_workers=args.nproc)
annotations = None
for converted_gts in all_converted_gts:
    converted_gts = [gt+"\n" for gt in converted_gts]
    if annotations is None:
        annotations = converted_gts
    else:
        annotations.extend(converted_gts)

with open(osp.join(args.dest_path, "annotations.txt"), "w") as f:
    f.writelines(annotations)
tqdm.write("done")