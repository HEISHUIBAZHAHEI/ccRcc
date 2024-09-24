from torchvision import transforms
import torch
import pandas as pd
from PIL import Image
import torch.utils.data as data
from torch.utils.data import Dataset
import random
from pathlib import Path
from variables import *
import numpy as np

class AddPepperNoise(object):

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) or (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255
            img_[mask == 2] = 0
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img

train_compose = transforms.Compose([
    transforms.Resize((image_size + padding_size, image_size + padding_size)),
    transforms.RandomCrop(crop_size),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(hue=0.5, contrast=0.5, brightness=0.5),
    transforms.RandomRotation(90),
    AddPepperNoise(0.1, p=0.15),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomErasing(p=0.4, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
])

test_compose = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class EVBSeT_CSV(Dataset):

    def __init__(self, root, type_):
        self.root = Path(root)
        self.type_ = type_
        if type_ == "train":
            self.csv = self.root / "carcinoma-of-kidney-train.csv"
            self.transform = train_compose
            # self.transform = test_compose
        elif type_ == "test":
            self.csv = self.root / "carcinoma-of-kidney-test.csv"
            self.transform = test_compose
        self.check_files(self.csv)
        try:
            self.csv = pd.read_csv(self.csv)
        except:
            self.csv = pd.read_csv(self.csv, encoding='gbk')
        self.csv = self.csv.dropna()
        self.csv['id'] = self.csv['id'].astype(str)
        self.people_classfiy = self.csv.loc[:, 'label'].map(lambda x: 1 if (x // 1) >= 3 else 0)
        # self.people_classfiy = self.csv.loc[:, 'label'].map(lambda x: 0 if )
        self.people_classfiy.index = self.csv['id']
        self.people_classfiy = self.people_classfiy.to_dict()

        self.neg_pic = []
        self.pos_pic = []
        self.pic_files = []
        for p in self.people_classfiy:
            if type_ == 'train':
                pic_file = self.root / str(p)
                pic_file = list(pic_file.rglob('*.bmp'))
            else:
                pic_file = self.root / str(p)
                pic_file = list(pic_file.rglob('*.bmp'))
            self.pic_files += pic_file

            if self.people_classfiy[p] == 1:
                self.pos_pic += pic_file
            else:
                self.neg_pic += pic_file

        if type_ == 'train':
            if len(self.pos_pic) >= len(self.neg_pic):
                ratio = int(len(self.pos_pic) // len(self.neg_pic))
                distance = len(self.pos_pic) - (ratio * len(self.neg_pic))
                self.neg_pic = ratio * self.neg_pic + self.neg_pic[0: distance]
                self.pic_files = self.pos_pic + self.neg_pic
                random.shuffle(self.pic_files)
            else:
                ratio = int(len(self.neg_pic) // len(self.pos_pic))
                distance = len(self.neg_pic) - (ratio * len(self.pos_pic))
                self.pos_pic = ratio * self.pos_pic + self.pos_pic[0: distance]
                self.pic_files = self.pos_pic + self.neg_pic
                random.shuffle(self.pic_files)
            # self.pic_files = self.pos_pic + self.neg_pic
        else:
            self.pic_files = self.pos_pic + self.neg_pic

    def check_files(self, file):
        print(Path(file))
        assert Path(file).exists(), FileExistsError('{str(file)}不存在')

    def __len__(self):
        return len(self.pic_files)

    def __getitem__(self, index):
        img_single = Image.open(str(self.pic_files[index]))
        people = str(self.pic_files[index].parent.name)
        level = self.csv.loc[self.csv['id'] == str(people), "label"].iloc[0]
        id = str(people)
        y = self.people_classfiy[str(people)]
        img_data = self.transform(img_single)
        rs = {
            "img": img_data,
            "label": torch.Tensor([y])[0],
            "id": id,
            "level": level,
            "image_path": str(self.pic_files[index])
        }
        return rs
