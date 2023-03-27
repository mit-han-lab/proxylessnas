import os
import os.path as osp
import numpy as np
from PIL import Image
import json
import shutil
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str, default="/data/junyanli/datasets/widerface/raw")
parser.add_argument("-o", type=str, default="/dev/shm/widerface_coco")
args = parser.parse_args()

dataset_root = args.i
destination_root = args.o


def make_dataset(labels_file, images_dir, output_json_file, output_images_dir):
    input = open(labels_file, "r")
    output = open(output_json_file, "w")
    data = {}
    for line in input.readlines():
        line = line.rstrip().split()
        if line[0] == "#":
            image_name = line[-1]
            data[image_name] = []
        else:
            label = list(map(float, line))
            bbox = np.array(label[:4])  # xywh
            landmark = np.zeros(10)
            if len(label) != 4:
                landmark[0] = label[4]    # l0_x
                landmark[1] = label[5]    # l0_y
                landmark[2] = label[7]    # l1_x
                landmark[3] = label[8]    # l1_y
                landmark[4] = label[10]   # l2_x
                landmark[5] = label[11]   # l2_y
                landmark[6] = label[13]   # l3_x
                landmark[7] = label[14]   # l3_y
                landmark[8] = label[16]   # l4_x
                landmark[9] = label[17]   # l4_y
            bbox = bbox.tolist()
            landmark = landmark.tolist()
            data[image_name].append({"bbox": bbox,
                                     "landmark": landmark})
    instances_json = dict(
        info=dict(
            year=2021,
            version=1.0,
            description="wider face in coco format",
            date_created=2021
        ),
        licenses=None,
        categories=[{"id": 1, "name": "face", "supercategory": "face"}],
        images=[],
        annotations=[]
    )

    image_id_cnt = 0
    annotation_id_cnt = 0
    valid_num = 0
    invalid_num = 0
    for image_name in tqdm(list(data.keys())):
        annotations = data[image_name]
        image_folder, image_name = image_name.split("/")
        image_full_path = osp.join(images_dir,
                                   image_folder,
                                   image_name)
        image = Image.open(image_full_path)
        base_scale = np.array((160, 128))
        bboxes = np.array([[annotation["bbox"][0], annotation["bbox"][1],
                            annotation["bbox"][0]+annotation["bbox"][2],
                            annotation["bbox"][1]+annotation["bbox"][3]] for annotation in annotations])

        useful = False
        need_crop = False
        for bbox in bboxes:
            bbox_w = bbox[2]-bbox[0]
            bbox_h = bbox[3]-bbox[1]
            area = bbox_w * bbox_h
            ratio = area/(base_scale[0]*base_scale[1])
            if 0.05 < ratio < 0.9 and bbox_w < base_scale[0]-5 and bbox_h < base_scale[1]-5:
                valid_num += 1
                need_crop = True
            elif ratio >= 2:
                useful = True
                continue
            else:
                invalid_num += 1
                continue

            if max(0, bbox[2]-base_scale[0]+1) >= min(image.width-base_scale[0]-1, bbox[0]-1) or max(0, bbox[3]-base_scale[1]+1) >= min(image.height-base_scale[1]-1, bbox[1]-1):
                valid_num -= 1
                invalid_num += 1
                continue
            x1 = np.random.randint(max(0, bbox[2]-base_scale[0]+1),
                                   min(image.width-base_scale[0]-1, bbox[0]-1))
            y1 = np.random.randint(max(0, bbox[3]-base_scale[1]+1),
                                   min(image.height-base_scale[1]-1, bbox[1]-1))
            x2 = x1 + base_scale[0]
            y2 = y1 + base_scale[1]
            assert x1 < bbox[0] and y1 < bbox[1] and x2 > bbox[2] and y2 > bbox[3]
            image_cropped = image.crop((x1, y1, x2, y2))
            assert image_cropped.width == 160 and image_cropped.height == 128
            cbboxes = contain_bboxes([x1, y1, x2, y2], bboxes)
            image_cropped_name = image_name.split(
                ".")[0] + f"_{x1}_{y1}_{x2}_{y2}." + image_name.split(".")[1]
            image_id_cnt += 1
            file_name = image_folder+"=="+image_cropped_name
            image_cropped.save(osp.join(output_images_dir, file_name))
            instances_json["images"].append(dict(
                date_captured="2021",
                file_name=file_name,
                id=image_id_cnt,
                height=image_cropped.height,
                width=image_cropped.width
            ))
            for cbbox in cbboxes:
                cbbox = np.array(cbbox) - np.array([x1, y1, x1, y1])
                assert (cbbox > 0).all(), cbboxes
                cbbox = cbbox.tolist()
                cbbox = [cbbox[0], cbbox[1],
                         cbbox[2]-cbbox[0],
                         cbbox[3]-cbbox[1]]
                annotation_id_cnt += 1
                instances_json["annotations"].append(dict(
                    id=annotation_id_cnt,
                    image_id=image_id_cnt,
                    category_id=1,
                    bbox=cbbox,
                    area=cbbox[-1]*cbbox[-2],
                    segmentation=[],
                    iscrowd=0
                ))
        if useful and not need_crop:
            height = image.height
            width = image.width
            image_id_cnt += 1
            file_name = image_folder+"=="+image_name
            shutil.copy(image_full_path, osp.join(output_images_dir, file_name))
            instances_json["images"].append(dict(
                date_captured="2021",
                file_name=file_name,
                id=image_id_cnt,
                height=height,
                width=width
            ))
            for annotation in annotations:
                annotation_id_cnt += 1
                _area = annotation['bbox'][-1]*annotation['bbox'][-2]
                valid_num += 1
                instances_json["annotations"].append(dict(
                    id=annotation_id_cnt,
                    image_id=image_id_cnt,
                    category_id=1,
                    bbox=annotation['bbox'],
                    area=annotation['bbox'][-1]*annotation['bbox'][-2],
                    segmentation=[],
                    iscrowd=0
                ))

    print(valid_num)
    print(valid_num/(invalid_num+valid_num))
    print(annotation_id_cnt)

    json.dump(instances_json, output, indent=2)
    input.close()
    output.close()


def contain_bboxes(region, bboxes):
    diff = bboxes - region
    contain_all = (diff[:, 0] > 0) & (diff[:, 1] > 0) & (diff[:, 2] < 0) & (diff[:, 3] < 0)
    return bboxes[contain_all.nonzero()]


train_images_dir = osp.join(dataset_root, 'WIDER_train', 'images')
val_images_dir = osp.join(dataset_root, 'WIDER_val', 'images')
train_labels_file = osp.join(dataset_root, "train/label.txt")
val_labels_file = osp.join(dataset_root, "val/label.txt")

os.makedirs(destination_root, exist_ok=True)
os.makedirs(osp.join(destination_root, "annotations"), exist_ok=True)
os.makedirs(osp.join(destination_root, "train2017"), exist_ok=True)
os.makedirs(osp.join(destination_root, "val2017"), exist_ok=True)
make_dataset(train_labels_file,
             train_images_dir,
             osp.join(destination_root, "annotations", "instances_train2017.json"),
             osp.join(destination_root, "train2017"))
make_dataset(val_labels_file,
             val_images_dir,
             osp.join(destination_root, "annotations", "instances_val2017.json"),
             osp.join(destination_root, "val2017"))
