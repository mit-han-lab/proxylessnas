import numpy as np
import cv2
import sys
sys.path.append('..')

from torch.utils import data
from torch.utils.data import DataLoader


def flip(img, annotation):
    img = np.fliplr(img).copy()
    h, w = img.shape[:2]

    x_min, y_min, x_max, y_max = annotation[0:4]
    landmark_x = annotation[4::2]
    landmark_y = annotation[4 + 1::2]

    bbox = np.array([w - x_max, y_min, w - x_min, y_max])
    for i in range(len(landmark_x)):
        landmark_x[i] = w - landmark_x[i]

    new_annotation = list()
    new_annotation.append(x_min)
    new_annotation.append(y_min)
    new_annotation.append(x_max)
    new_annotation.append(y_max)

    for i in range(len(landmark_x)):
        new_annotation.append(landmark_x[i])
        new_annotation.append(landmark_y[i])

    return img, new_annotation


def channel_shuffle(img, annotation):
    if (img.shape[2] == 3):
        ch_arr = [0, 1, 2]
        np.random.shuffle(ch_arr)
        img = img[..., ch_arr]
    return img, annotation


def random_noise(img, annotation, limit=[0, 0.2], p=0.5):
    if random.random() < p:
        H, W = img.shape[:2]
        noise = np.random.uniform(limit[0], limit[1], size=(H, W)) * 255

        img = img + noise[:, :, np.newaxis] * np.array([1, 1, 1])
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img, annotation


def random_brightness(img, annotation, brightness=0.3):
    alpha = 1 + np.random.uniform(-brightness, brightness)
    img = alpha * image
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, annotation


def random_contrast(img, annotation, contrast=0.3):
    coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
    alpha = 1.0 + np.random.uniform(-contrast, contrast)
    gray = img * coef
    gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    img = alpha * img + gray
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, annotation


def random_saturation(img, annotation, saturation=0.5):
    coef = nd.array([[[0.299, 0.587, 0.114]]])
    alpha = np.random.uniform(-saturation, saturation)
    gray = img * coef
    gray = np.sum(gray, axis=2, keepdims=True)
    img = alpha * img + (1.0 - alpha) * gray
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, annotation


def random_hue(image, annotation, hue=0.5):
    h = int(np.random.uniform(-hue, hue) * 180)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + h) % 180
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image, annotation


def scale(img, annotation):
    f_xy = np.random.uniform(-0.4, 0.8)
    origin_h, origin_w = img.shape[:2]

    bbox = annotation[0:4]
    landmark_x = annotation[4::2]
    landmark_y = annotation[4 + 1::2]

    h, w = int(origin_h * f_xy), int(origin_w * f_xy)
    image = resize(img, (h, w),
                   preserve_range=True,
                   anti_aliasing=True,
                   mode='constant').astype(np.uint8)

    new_annotation = list()
    for i in range(len(bbox)):
        bbox[i] = bbox[i] * f_xy
        new_annotation.append(bbox[i])

    for i in range(len(landmark_x)):
        landmark_x[i] = landmark_x[i] * f_xy
        landmark_y[i] = landmark_y[i] * f_xy
        new_annotation.append(landmark_x[i])
        new_annotation.append(landmark_y[i])

    return image, new_annotation


def rotate(img, annotation, alpha=30):

    bbox = annotation[0:4]
    landmark_x = annotation[4::2]
    landmark_y = annotation[4 + 1::2]
    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat,
                                          (img.shape[1], img.shape[0]))

    point_x = [bbox[0], bbox[2], bbox[0], bbox[2]]
    point_y = [bbox[1], bbox[3], bbox[3], bbox[1]]

    new_point_x = list()
    new_point_y = list()
    for (x, y) in zip(landmark_x, landmark_y):
        new_point_x.append(rot_mat[0][0] * x + rot_mat[0][1] * y +
                           rot_mat[0][2])
        new_point_y.append(rot_mat[1][0] * x + rot_mat[1][1] * y +
                           rot_mat[1][2])

    new_annotation = list()
    new_annotation.append(min(new_point_x))
    new_annotation.append(min(new_point_y))
    new_annotation.append(max(new_point_x))
    new_annotation.append(max(new_point_y))

    for (x, y) in zip(landmark_x, landmark_y):
        new_annotation.append(rot_mat[0][0] * x + rot_mat[0][1] * y +
                              rot_mat[0][2])
        new_annotation.append(rot_mat[1][0] * x + rot_mat[1][1] * y +
                              rot_mat[1][2])

    return img_rotated_by_alpha, new_annotation


class WLFWDatasets(data.Dataset):
    def __init__(self, file_list, landmark, transforms=None):
        self.line = None
        self.path = None
        self.landmarks = None
        self.attribute = None
        self.filenames = None
        self.euler_angle = None
        self.transforms = transforms
        self.landmark_num = landmark
        with open(file_list, 'r') as f:
            self.lines = f.readlines()

    def __getitem__(self, index):
        self.line = self.lines[index].strip().split()
        self.img = cv2.imread(self.line[0])
        self.landmark = np.asarray(self.line[1:1+self.landmark_num*2], dtype=np.float32)
        self.attribute = np.asarray(self.line[1+self.landmark_num*2:-3], dtype=np.int32)
        self.euler_angle = np.asarray(self.line[-3:], dtype=np.float32)
        if self.transforms:
            self.img = self.transforms(self.img)
        return (self.img, self.landmark, self.attribute, self.euler_angle)

    def __len__(self):
        return len(self.lines)


if __name__ == '__main__':
    file_list = './data/test_data/list.txt'
    wlfwdataset = WLFWDatasets(file_list)
    dataloader = DataLoader(wlfwdataset,
                            batch_size=256,
                            shuffle=True,
                            num_workers=0,
                            drop_last=False)
    for img, landmark, attribute, euler_angle in dataloader:
        print("img shape", img.shape)
        print("landmark size", landmark.size())
        print("attrbute size", attribute)
        print("euler_angle", euler_angle.size())
