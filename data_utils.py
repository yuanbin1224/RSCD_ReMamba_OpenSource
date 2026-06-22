import os
import random
from os.path import join

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".TIF", ".TIFF", ".PNG", ".JPG", ".JPEG", ".BMP")


def is_image_file(filename):
    return filename.endswith(IMAGE_SUFFIXES)


def list_image_names(image_dir, suffixes=None):
    if suffixes:
        expanded = []
        for suffix in suffixes:
            expanded.extend([suffix, suffix.lower(), suffix.upper()])
        suffixes = tuple(set(expanded))
    else:
        suffixes = IMAGE_SUFFIXES
    return [name for name in sorted(os.listdir(image_dir)) if name.endswith(suffixes) and is_image_file(name)]


def image_transform(normalize=True):
    ops = [transforms.ToTensor()]
    if normalize:
        ops.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(ops)


def load_binary_mask(path):
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    mask = (mask > 0).astype(np.float32)
    return torch.from_numpy(mask).unsqueeze(0)


def calMetric_iou(gt_value, result):
    gt_bool = np.asarray(gt_value) > 0
    res_bool = np.asarray(result) > 0
    intersection = np.logical_and(gt_bool, res_bool).sum()
    union = np.logical_or(gt_bool, res_bool).sum()
    return float(intersection), float(union)


def calMetric_somemetric(predict, label):
    predict = np.asarray(predict) > 0
    label = np.asarray(label) > 0
    tp = np.logical_and(predict, label).sum()
    tn = np.logical_and(~predict, ~label).sum()
    fp = np.logical_and(predict, ~label).sum()
    fn = np.logical_and(~predict, label).sum()
    return int(tp), int(tn), int(fp), int(fn)


def visualize_metrics(label, predict):
    label = np.asarray(label) > 0
    predict = np.asarray(predict) > 0
    output = np.zeros((*label.shape, 3), dtype=np.uint8)
    output[np.logical_and(predict, label)] = [255, 255, 255]
    output[np.logical_and(predict, ~label)] = [255, 0, 0]
    output[np.logical_and(~predict, label)] = [0, 255, 0]
    return output


class PairAugmentation:
    def __init__(self, crop=True, crop_size=512, augment=True, angle=30):
        self.crop = crop
        self.crop_size = crop_size
        self.augment = augment
        self.angle = angle

    def _crop(self, image1, image2, mask):
        width, height = image1.size
        crop_w = min(self.crop_size, width)
        crop_h = min(self.crop_size, height)
        if width == crop_w:
            left = 0
        else:
            left = random.randint(0, width - crop_w)
        if height == crop_h:
            top = 0
        else:
            top = random.randint(0, height - crop_h)
        box = (left, top, left + crop_w, top + crop_h)
        return image1.crop(box), image2.crop(box), mask.crop(box)

    def __call__(self, image1, image2, mask):
        if self.crop:
            image1, image2, mask = self._crop(image1, image2, mask)

        if self.augment:
            prop = random.random()
            if prop < 0.15:
                image1 = image1.transpose(Image.FLIP_LEFT_RIGHT)
                image2 = image2.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            elif prop < 0.30:
                image1 = image1.transpose(Image.FLIP_TOP_BOTTOM)
                image2 = image2.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            elif prop < 0.50:
                angle = random.uniform(-self.angle, self.angle)
                image1 = image1.rotate(angle, resample=Image.BILINEAR)
                image2 = image2.rotate(angle, resample=Image.BILINEAR)
                mask = mask.rotate(angle, resample=Image.NEAREST)

        return image1, image2, mask


class LoadDatasetFromFolder(Dataset):
    def __init__(self, args, hr1_path, hr2_path, lab_path):
        super().__init__()
        self.names = list_image_names(hr1_path, getattr(args, "suffix", None))
        self.hr1_filenames = [join(hr1_path, name) for name in self.names]
        self.hr2_filenames = [join(hr2_path, name) for name in self.names]
        self.lab_filenames = [join(lab_path, name) for name in self.names]
        self.transform = image_transform(normalize=True)

    def __getitem__(self, index):
        image1 = self.transform(Image.open(self.hr1_filenames[index]).convert("RGB"))
        image2 = self.transform(Image.open(self.hr2_filenames[index]).convert("RGB"))
        label = load_binary_mask(self.lab_filenames[index])
        return image1, image2, label

    def __len__(self):
        return len(self.hr1_filenames)


class DA_DatasetFromFolder(Dataset):
    def __init__(
        self,
        image_dir1,
        image_dir2,
        label_dir,
        crop=True,
        crop_size=512,
        augment=True,
        angle=30,
        suffixes=None,
    ):
        super().__init__()
        self.names = list_image_names(image_dir1, suffixes)
        self.image_filenames1 = [join(image_dir1, name) for name in self.names]
        self.image_filenames2 = [join(image_dir2, name) for name in self.names]
        self.label_filenames = [join(label_dir, name) for name in self.names]
        self.augment = PairAugmentation(crop=crop, crop_size=crop_size, augment=augment, angle=angle)
        self.transform = image_transform(normalize=True)

    def __getitem__(self, index):
        image1 = Image.open(self.image_filenames1[index]).convert("RGB")
        image2 = Image.open(self.image_filenames2[index]).convert("RGB")
        label = Image.open(self.label_filenames[index]).convert("L")
        image1, image2, label = self.augment(image1, image2, label)

        image1 = self.transform(image1)
        image2 = self.transform(image2)
        label = torch.from_numpy((np.asarray(label, dtype=np.uint8) > 0).astype(np.float32)).unsqueeze(0)
        return image1, image2, label, self.names[index]

    def __len__(self):
        return len(self.image_filenames1)


class TestDatasetFromFolder(Dataset):
    def __init__(self, args, time1_dir, time2_dir, label_dir):
        super().__init__()
        self.names = list_image_names(time1_dir, getattr(args, "suffix", None))
        self.image1_filenames = [join(time1_dir, name) for name in self.names]
        self.image2_filenames = [join(time2_dir, name) for name in self.names]
        self.label_filenames = [join(label_dir, name) for name in self.names]
        self.transform = image_transform(normalize=True)

    def __getitem__(self, index):
        image1 = self.transform(Image.open(self.image1_filenames[index]).convert("RGB"))
        image2 = self.transform(Image.open(self.image2_filenames[index]).convert("RGB"))
        label = load_binary_mask(self.label_filenames[index])
        return image1, image2, label, self.names[index]

    def __len__(self):
        return len(self.image1_filenames)


class PredictDatasetFromFolder(Dataset):
    def __init__(self, args, time1_dir, time2_dir):
        super().__init__()
        self.names = list_image_names(time1_dir, getattr(args, "suffix", None))
        self.image1_filenames = [join(time1_dir, name) for name in self.names]
        self.image2_filenames = [join(time2_dir, name) for name in self.names]
        self.transform = image_transform(normalize=True)

    def __getitem__(self, index):
        image1 = self.transform(Image.open(self.image1_filenames[index]).convert("RGB"))
        image2 = self.transform(Image.open(self.image2_filenames[index]).convert("RGB"))
        return image1, image2, self.names[index]

    def __len__(self):
        return len(self.image1_filenames)
