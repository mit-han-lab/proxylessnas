import os
import os.path as osp
import argparse
import random
from tqdm import trange, tqdm
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Split dataset')
parser.add_argument('-i', type=str, required=True)
parser.add_argument('-o', type=str, required=True)
parser.add_argument('-dataset', type=str, choices=['mpiigaze'], default='mpiigaze')
parser.add_argument('--deploy', action="store_true")
args = parser.parse_args()

with open(args.i, "r") as f:
    all_samples = f.readlines()

if args.deploy:
    random.shuffle(all_samples)
    test_samples   = all_samples[:int(len(all_samples)*0.1)]
    train_samples = all_samples[int(len(all_samples)*0.1):]
    with open(osp.join(args.o, 'train.txt'), "w") as f:
        f.writelines(train_samples)
    with open(osp.join(args.o, 'test.txt'), "w") as f:
        f.writelines(test_samples)
    with open(osp.join(args.o, 'val.txt'), "w") as f:
        f.writelines(test_samples)
    tqdm.write(f"{len(train_samples)} train | {len(test_samples)} test")
else:
    if args.dataset == 'mpiigaze':
        subjects = {}
        for i in range(15):
            id = 'p'+str(i).zfill(2)
            subjects[id] = []
            for sample in all_samples:
                if id in sample:
                    subjects[id].append(sample)
        for i in trange(15):
            id = 'p'+str(i).zfill(2)
            test_samples = subjects[id]
            train_samples = []
            for j in range(15):
                id_2 = 'p'+str(j).zfill(2)
                if i == j:
                    continue
                train_samples.extend(subjects[id_2])
            random.shuffle(train_samples)
            # val_samples = train_samples[:int(len(train_samples)*0.15)]
            # train_samples = train_samples[int(len(train_samples)*0.15):]
            os.makedirs(osp.join(args.o, f"fold_{i}"), exist_ok=True)
            with open(osp.join(args.o, f"fold_{i}", 'train.txt'), "w") as f:
                f.writelines(train_samples)
            with open(osp.join(args.o, f"fold_{i}", 'test.txt'), "w") as f:
                f.writelines(test_samples)
            with open(osp.join(args.o, f"fold_{i}", 'val.txt'), "w") as f:
                f.writelines(test_samples)
            tqdm.write(f"fold {i}: {len(train_samples)} train | {len(test_samples)} test")
    else:
        raise NotImplementedError

print("done") 