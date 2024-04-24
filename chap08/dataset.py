import io
import os
import json
import random

import torch
import torchvision.transforms as T
from torchvision.transforms import v2
from torchvision.io import read_image, ImageReadMode

from PIL import Image
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import FileLister, FileOpener
from tokenizer import tokenize


def build_preprocess(img_size=224):
    to_rgb = [T.Lambda(lambda x: x.convert("RGB"))]

    resized_crop = [
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),]
    norm = [
        T.ToTensor(),
        T.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),),
        ]
    return T.Compose([*resized_crop, *to_rgb, *norm])

def apply_transform(item, image_transform=None, text_transform=None):
    def decode_img(stream):
        img = Image.open(io.BytesIO(stream)).convert("RGB")
        if image_transform is not None:
            img = image_transform(img)
        return img

    def decode_txt(stream):
        txt = stream.decode("utf-8")
        if text_transform is not None:
            txt = text_transform(txt)
        return txt

    img = decode_img(item[".jpg"].read())
    txt = decode_txt(item[".txt"].read())
    return img, txt

def build_datapipe(cfg, image_transform, text_transform, shuffle, num_shards, rank):
    ds_root=cfg.dataset.gcc12m.path
    ds_length = cfg.dataset.gcc12m.length
    dp = FileLister(ds_root, "*.tar", recursive=True)
    dp = FileOpener(dp, mode="b")
    dp = dp.load_from_tar(length=ds_length)
    dp = dp.webdataset()

    if shuffle:
        dp = dp.shuffle()
    
    dp = dp.sharding_filter()
    dp.apply_sharding(num_shards, rank)
    dp = dp.map(lambda x: apply_transform(x, image_transform, text_transform))
    return dp


def build_dataloader(cfg, local_rank, world_size):
    if world_size == 1:
        # for single GPU
        dp = build_datapipe(cfg=cfg,
                            image_transform=build_preprocess(cfg.dataset.gcc12m.img_size),
                            text_transform=tokenize,
                            shuffle=True,
                            num_shards=1,
                            rank=0)
    else:
        # for multiple GPUs
        dp = build_datapipe(cfg=cfg,
                            image_transform=build_preprocess(cfg.dataset.gcc12m.img_size),
                            text_transform=tokenize,
                            shuffle=True,
                            num_shards=world_size,
                            rank=local_rank)

    dataloader = DataLoader(
        dp,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader


## for ImageNet1K

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
        
    except FileNotFoundError:
        print(f"The file at path {file_path} does not exist.")
    except json.JSONDecodeError:
        print(f"The file at path {file_path} is not a valid JSON file.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
def get_IN1K_transform_val(resized_size=256, img_size=224):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(size=(resized_size, resized_size), antialias=True),
        v2.CenterCrop(size=(img_size, img_size)),
        v2.ToDtype(torch.float32, scale=True),  # convert [0, 255] to [0.0, 1.0]
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.ToPureTensor(),
    ])


class ImageNet1K(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        self.class_idx = read_json_file(config.dataset.in1k.class_idx_path)
        self.idx2label = [self.class_idx[str(k)][1] for k in range(len(self.class_idx))]

        self.img_size = config.dataset.in1k.img_size
        resized_img_size = int(self.img_size * 1.145)
        self.transform = get_IN1K_transform_val(resized_size=resized_img_size, img_size=self.img_size)
        self.classes, self.class_to_idx = self._find_classes(config.dataset.in1k.val_path)
        self.samples = self._make_dataset(config.dataset.in1k.val_path)
        
    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        :return: Two lists: list of class names and corresponding class indices.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self, img_dir):
        """
        Creates a list of samples with their class indices.
        """
        samples = []
        for target_class in sorted(self.class_to_idx.keys()):
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(img_dir, target_class)
            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    if self._is_image_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, class_index)
                        samples.append(item)
        return samples

    def _is_image_file(self, filename):
        """Checks if a file is an image."""
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        random_idx = random.randint(0, len(self))
        
        image_path, target = self.samples[idx]
        try:
            img = read_image(image_path, mode=ImageReadMode.RGB)
        except:
            print(f"{image_path} doesn't exist!")
            return self.__getitem__(random_idx)

        img = self.transform(img)
        return img, target