#-*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import shutil
import sys
from tqdm.contrib.concurrent import process_map
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from pfld.utils import calculate_pitch_yaw_roll
debug = False

convert_106_to_98_list = list(range(0,55))+[58,59,60,61,62]+list(range(66,74))+list(range(75,83))+list(range(84,104))+[104,105]
assert len(convert_106_to_98_list) == 98

def rotate(angle, center, landmark):
    rad = angle * np.pi / 180.0
    alpha = np.cos(rad)
    beta = np.sin(rad)
    M = np.zeros((2,3), dtype=np.float32)
    M[0, 0] = alpha
    M[0, 1] = beta
    M[0, 2] = (1-alpha)*center[0] - beta*center[1]
    M[1, 0] = -beta
    M[1, 1] = alpha
    M[1, 2] = beta*center[0] + (1-alpha)*center[1]

    landmark_ = np.asarray([(M[0,0]*x+M[0,1]*y+M[0,2],
                             M[1,0]*x+M[1,1]*y+M[1,2]) for (x,y) in landmark])
    return M, landmark_

class ImageData():
    def __init__(self, line, imgDir, image_size=112):
        if "\x00" in line: # TODO
            self.empty = True
            return
        self.image_size = image_size
        line = line.strip().split()
        
        #0-195: landmark 坐标点  196-199: bbox 坐标点;
        #200: 姿态(pose)         0->正常姿态(normal pose)          1->大的姿态(large pose)
        #201: 表情(expression)   0->正常表情(normal expression)    1->夸张的表情(exaggerate expression)
        #202: 照度(illumination) 0->正常照明(normal illumination)  1->极端照明(extreme illumination)
        #203: 化妆(make-up)      0->无化妆(no make-up)             1->化妆(make-up)
        #204: 遮挡(occlusion)    0->无遮挡(no occlusion)           1->遮挡(occlusion)
        #205: 模糊(blur)         0->清晰(clear)                    1->模糊(blur)
        #206: 图片名称
        if not (len(line[-1].split(',')) == 106*2):
            self.empty = True
            return
        self.landmark = np.array(list(map(int, line[2].split(",")))).reshape(-1, 2)
        self.landmark = self.landmark[convert_106_to_98_list]
        self.box = np.array(list(map(float, line[1].split(",")))).astype(int)
        self.pose = False
        self.expression = False
        self.illumination = False
        self.make_up = False
        self.occlusion = False
        self.blur = False
        name = line[0].split("_")
        name = " ".join(name[:-1])+"_"+name[-1]
        self.path = imgDir.rstrip("/") + "/" + name.lstrip("/")
        self.img = None
        self.imgs = []
        self.landmarks = []
        self.boxes = []
        self.empty = False
        if not (self.landmark >= 0).all():
            self.empty = True

    def load_data(self, is_train, repeat, mirror=None):
        assert (self.landmark >= 0).all(), str(self.landmark)
        
        if (mirror is not None):
            with open(mirror, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 1
                mirror_idx = lines[0].strip().split(',')
                mirror_idx = list(map(int, mirror_idx))
        xy = np.min(self.landmark, axis=0).astype(np.int32) 
        zz = np.max(self.landmark, axis=0).astype(np.int32)
        wh = zz - xy + 1

        center = (xy + wh/2).astype(np.int32)
        img = cv2.imread(self.path)
        if img is None:
            self.empty = True
            return
            
        boxsize = int(np.max(wh)*1.2)
        xy = center - boxsize//2
        x1, y1 = xy
        x2, y2 = xy + boxsize
        height, width, _ = img.shape
        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)

        imgT = img[y1:y2, x1:x2]
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

        imgT = cv2.resize(imgT, (self.image_size, self.image_size))
        landmark = (self.landmark - xy)/boxsize
        assert (landmark >= 0).all(), str(landmark) + str([dx, dy])
        assert (landmark <= 1).all(), str(landmark) + str([dx, dy])
        self.imgs.append(imgT)
        self.landmarks.append(landmark)

        if is_train:
            while len(self.imgs) < repeat:
                angle = np.random.randint(-30, 30)
                cx, cy = center
                cx = cx + int(np.random.randint(-boxsize*0.1, boxsize*0.1))
                cy = cy + int(np.random.randint(-boxsize * 0.1, boxsize * 0.1))
                M, landmark = rotate(angle, (cx,cy), self.landmark)

                imgT = cv2.warpAffine(img, M, (int(img.shape[1]*1.1), int(img.shape[0]*1.1)))

                
                wh = np.ptp(landmark, axis=0).astype(np.int32) + 1
                size = np.random.randint(int(np.min(wh)), np.ceil(np.max(wh) * 1.25))
                xy = np.asarray((cx - size // 2, cy - size//2), dtype=np.int32)
                landmark = (landmark - xy) / size
                if (landmark < 0).any() or (landmark > 1).any():
                    continue

                x1, y1 = xy
                x2, y2 = xy + size
                height, width, _ = imgT.shape
                dx = max(0, -x1)
                dy = max(0, -y1)
                x1 = max(0, x1)
                y1 = max(0, y1)

                edx = max(0, x2 - width)
                edy = max(0, y2 - height)
                x2 = min(width, x2)
                y2 = min(height, y2)

                imgT = imgT[y1:y2, x1:x2]
                if (dx > 0 or dy > 0 or edx >0 or edy > 0):
                    imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

                imgT = cv2.resize(imgT, (self.image_size, self.image_size))

                if mirror is not None and np.random.choice((True, False)):
                    landmark[:,0] = 1 - landmark[:,0]
                    landmark = landmark[mirror_idx]
                    imgT = cv2.flip(imgT, 1)
                self.imgs.append(imgT)
                self.landmarks.append(landmark)

    def save_data(self, path, prefix):
        attributes = [self.pose, self.expression, self.illumination, self.make_up, self.occlusion, self.blur]
        attributes = np.asarray(attributes, dtype=np.int32)
        attributes_str = ' '.join(list(map(str, attributes)))
        labels = []
        # 98
        TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
        # 106
        # TRACKED_POINTS = [33, 38, 50, 46, 66, 70, 75, 79, 58, 62, 84, 90, 93, 16]
        for i, (img, lanmark) in enumerate(zip(self.imgs, self.landmarks)):
            assert lanmark.shape == (98, 2)
            save_path = os.path.join(path, prefix+'_'+str(i)+'.png')
            assert not os.path.exists(save_path), save_path
            cv2.imwrite(save_path, img)

            euler_angles_landmark = []
            for index in TRACKED_POINTS:
                euler_angles_landmark.append(lanmark[index])
            euler_angles_landmark = np.asarray(euler_angles_landmark).reshape((-1, 28))
            pitch, yaw, roll = calculate_pitch_yaw_roll(euler_angles_landmark[0])
            euler_angles = np.asarray((pitch, yaw, roll), dtype=np.float32)
            euler_angles_str = ' '.join(list(map(str, euler_angles)))

            landmark_str = ' '.join(list(map(str,lanmark.reshape(-1).tolist())))

            label = '{} {} {} {}\n'.format(save_path, landmark_str, attributes_str, euler_angles_str)

            labels.append(label)
        return labels
    

def handle_single(attr):
    i, line, imgDir, outDir, is_train = attr
    save_img = os.path.join(outDir, 'imgs')
    Img = ImageData(line, imgDir)
    if Img.empty:
        return ""
    img_name = Img.path
    Img.load_data(is_train, 10, "Mirror98.txt")
    if Img.empty:
        return ""
    _, filename = os.path.split(img_name)
    filename, _ = os.path.splitext(filename)
    filename = filename.replace(" ", "+")
    label_txt = Img.save_data(save_img, str(i)+'_' + filename)
    return label_txt

def get_dataset_list(imgDir, outDir, landmarkDir, is_train):
    with open(landmarkDir, 'r') as f:
        lines = f.readlines()
        labels = []
        save_img = os.path.join(outDir, 'imgs')
        if not os.path.exists(save_img):
            os.mkdir(save_img)

        if debug:
            lines = lines[:100]
        
        lines = list(zip(range(len(lines)), lines, 
                         [imgDir]*len(lines), 
                         [outDir]*len(lines), 
                         [is_train]*len(lines)))
        labels = process_map(handle_single, lines, chunksize=10)

    with open(os.path.join(outDir, 'list.txt'), 'w') as f:
        for label in labels:
            if label is not None:
                f.writelines(label)

if __name__ == '__main__':
    root_dir = "/dev/shm/facescrub_ready"
    landmarkDirs = ['/data/junyanli/datasets/megaface/facescrub_face_info.txt']
    imageDirs = ['/data/junyanli/datasets/megaface/data/FaceScrub']
    outDirs = ['train_data']
    for landmarkDir, outDir, is_train, imageDir in zip(landmarkDirs, outDirs, [True], imageDirs):
        outDir = os.path.join(root_dir, outDir)
        print(outDir)
        if os.path.exists(outDir):
            shutil.rmtree(outDir)
        os.mkdir(outDir)
        get_dataset_list(imageDir, outDir, landmarkDir, is_train)
    print('done')