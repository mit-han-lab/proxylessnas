import os
import sys
from tqdm import tqdm

def check(img_dir, anno_filename, type):
    cnt = 0
    with open(anno_filename, "r") as f:
        samples = f.readlines()
    for sample in tqdm(samples):
        sample = sample.strip().split()
        name = sample[0]
        if type == "FaceScrub":
            folder, filename = name.split("/")
            folder = folder.replace("_", " ")
            filename = filename.split("_")
            filename = " ".join(filename[:-1]) + "_" + filename[-1]
            full_path = os.path.join(img_dir, folder, filename)
        elif type == "MegaFaceDistractor":
            full_path = img_dir.rstrip("/") + "/" + name
        if not os.path.exists(full_path):
            cnt += 1
            tqdm.write(f"{cnt} {full_path}")
    return cnt / len(samples)

rate1 = check("/data/junyanli/datasets/megaface/data/FaceScrub", "/data/junyanli/datasets/megaface/facescrub_face_info.txt", type="FaceScrub")
rate2 = check("/data/junyanli/datasets/megaface/data/MegaFaceDistractor/FlickrFinal2", "/data/junyanli/datasets/megaface/megaface_face_info.txt", type="MegaFaceDistractor")

print(rate1)
print(rate2)