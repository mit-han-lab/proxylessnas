import torch
import torchvision
import random
import os
import shutil
import argparse
import os.path as osp
import models
from PIL import Image
import cv2
import numpy as np
from tqdm import trange, tqdm
import platform
OS_PLATFORM = platform.platform()

def euler_to_vec(theta, phi):
    x = -1 * np.cos(theta) * np.sin(phi)
    y = -1 * np.sin(theta)
    z = -1 * np.cos(theta) * np.cos(phi)
    vec = np.array([x, y, z])
    vec = vec / np.linalg.norm(vec)
    return vec

def vec_to_euler(x,y,z):
    theta = np.arcsin(-y)
    phi = np.arctan2(-x, -z)
    return theta, phi

def calc_angle_error(pred, gt):
    pred_vec = euler_to_vec(pred[0], pred[1])
    gt_vec = euler_to_vec(gt[0], gt[1])
    error = np.rad2deg(np.arccos(np.dot(pred_vec, gt_vec)))
    return error

# def draw_gaze(image_in, eye_pos, pitchyaw, length=150.0, thickness=2,
#               color=(0,0,255)):
#     """Draw gaze angle on given image with a given eye positions."""
#     image_out = image_in
#     if len(image_out.shape) == 2 or image_out.shape[2] == 1:
#         image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
#     dx = -length * np.sin(pitchyaw[1])
#     dy = -length * np.sin(pitchyaw[0])
#     cv2.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
#                    tuple(np.round([eye_pos[0] + dx,
#                                    eye_pos[1] + dy]).astype(int)), color,
#                    thickness, cv2.LINE_AA, tipLength=0.2)
#     return image_out

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


def str_gaze(gaze, deg=True):
    # pitch yaw (theta phi)
    from copy import copy
    _gaze = copy(gaze)
    if deg:
        _gaze = np.rad2deg(gaze)
    return f"yaw: {_gaze[1]:.2f}    pitch: {_gaze[0]:.2f}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', required=True, type=str)
    parser.add_argument('-ckpt', required=True, type=str)
    parser.add_argument('-file', required=True, type=str)
    parser.add_argument('-arch', required=True, type=str)
    parser.add_argument('-num', required=False, type=int)
    parser.add_argument('--all', action="store_true")
    args = parser.parse_args()

    shutil.rmtree("examples", ignore_errors=True)
    os.makedirs("examples")

    with open(args.file, "r") as f:
        annos = f.readlines()

    model = getattr(models, args.model)(arch=args.arch)
    ckpt = torch.load(args.ckpt)
    state_dict = ckpt['state_dict']
    keys = list(state_dict.keys())
    for key in keys:
        state_dict[key.replace("model.","")] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.eval().cuda()
    

    transform = torchvision.transforms.ToTensor()
    
    filenames = []
    gt_pitchyaws = []
    pred_pitchyaws = []
    errors = []
    
    random.shuffle(annos)
    N = len(annos) if args.all else args.num

    for i in trange(N):
        anno = annos[i].strip().split()
        leye = transform(Image.open(anno[0])).cuda().unsqueeze(0)
        reye = transform(Image.open(anno[1])).cuda().unsqueeze(0)
        face = transform(Image.open(anno[2])).cuda().unsqueeze(0)
        with torch.no_grad():
            pred_pitchyaw = model(leye, reye, face) / 100.0
            pred_pitchyaw = pred_pitchyaw.detach().cpu().numpy()[0].tolist()

        if "Windows" in OS_PLATFORM:
            filename = "_".join(anno[2].split("\\")[-2:])
        else:
            filename = "_".join(anno[2].split("/")[-2:])
        img = cv2.imread(anno[2])
        gt_pitchyaw = list(map(float, anno[3:]))
        # color: BGR
        to_visualize = draw_gaze(img, (img.shape[1]//2, img.shape[0]//4), pred_pitchyaw, color=(0,255,0))
        to_visualize = draw_gaze(to_visualize, (img.shape[1]//2, img.shape[0]//4), gt_pitchyaw, color=(0,0,255))
        
        error = calc_angle_error(pred_pitchyaw, gt_pitchyaw)
        tqdm.write("="*50)
        tqdm.write(filename)
        tqdm.write(f'pred: {str_gaze(pred_pitchyaw)}')
        tqdm.write(f'gt:   {str_gaze(gt_pitchyaw)}')
        tqdm.write(f'error: {error}')
        tqdm.write("="*50)
        cv2.imwrite(osp.join("examples", filename.replace("face", f"{error:.2f}")), to_visualize)
        
        filenames.append(filename)
        gt_pitchyaws.append(gt_pitchyaw)
        pred_pitchyaws.append(pred_pitchyaw)
        errors.append(error)
    np.savez("./examples/result.npz", filenames=filenames, gt_pitchyaws=gt_pitchyaws, pred_pitchyaws=pred_pitchyaws)
    print("error mean:", np.mean(errors))
    print("error std:", np.std(errors))
    

