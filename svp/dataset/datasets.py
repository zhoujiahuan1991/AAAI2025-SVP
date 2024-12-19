# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from .fgvc import CUB200Dataset, NabirdsDataset, DogsDataset, FlowersDataset, CarsDataset
from .dtd import DTD
from .gtsrb import GTSRB
from .food import Food101
from .svhn import SVHN
from .vtab import VTAB


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if is_train == True:
        split = 'train'
    elif is_train == False:
        split = "test"

    # 指定你希望数据集被下载和存储的目录
    desired_download_root = '/data/dataset/yaoyifeng/'
    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(root=desired_download_root, train=is_train, transform=transform, download=True)
        nb_classes = 100
    if args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(root=desired_download_root, train=is_train, transform=transform, download=True)
        nb_classes = 10
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == "CUB200":
        dataset = CUB200Dataset(args.data_path, split=split, transform=transform)
        nb_classes = dataset.get_class_num()
    elif args.data_set == "NABIRDS":
        dataset = NabirdsDataset(args.data_path, split=split, transform=transform)
        nb_classes = dataset.get_class_num()
    elif args.data_set == "DOGS":
        dataset = DogsDataset(args.data_path, split=split, transform=transform)
        nb_classes = dataset.get_class_num()
    elif args.data_set == "FLOWERS":
        dataset = FlowersDataset(args.data_path, split=split, transform=transform)
        nb_classes = dataset.get_class_num()
    elif args.data_set == "CARS":
        dataset = CarsDataset(args.data_path, split=split, transform=transform)
        nb_classes = dataset.get_class_num()
    elif args.data_set == "DTD":
        dataset = DTD(data_path=args.data_path, split=split, transform=transform)
        nb_classes = dataset.get_class_num()
    elif args.data_set == "GTSRB":
        dataset = GTSRB(data_path=args.data_path, split=split, transform=transform)
        nb_classes = dataset.get_class_num()
    elif args.data_set == "FOOD":
        dataset = Food101(data_path=args.data_path, split=split, transform=transform)
        nb_classes = dataset.get_class_num()
    elif args.data_set == "SVHN":
        dataset = SVHN(data_path=args.data_path, split=split, transform=transform)
        nb_classes = dataset.get_class_num()

    elif args.data_set == "VTAB_CIFAR100":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "cifar")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == "VTAB_Caltech101":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "caltech101")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 102
    elif args.data_set == "VTAB_DTD":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "dtd")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 47
    elif args.data_set == "VTAB_Flowers102":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "oxford_flowers102")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 102
    elif args.data_set == "VTAB_Pets":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "oxford_iiit_pet")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 37
    elif args.data_set == "VTAB_SVNH":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "svhn")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 10
    elif args.data_set == "VTAB_Sun397":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "sun397")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 397
    elif args.data_set == "VTAB_Camelyon":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "patch_camelyon")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 2
    elif args.data_set == "VTAB_EuroSAT":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "eurosat")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 10
    elif args.data_set == "VTAB_Resisc45":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "resisc45")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 45
    elif args.data_set == "VTAB_Retinopathy":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "diabetic_retinopathy")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 5
    elif args.data_set == "VTAB_Clevr_Count":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "clevr_count")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 8
    elif args.data_set == "VTAB_Clevr_Dist":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "clevr_dist")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 6
    elif args.data_set == "VTAB_DMLab":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "dmlab")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 6
    elif args.data_set == "VTAB_KITTI_Dist":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "kitti")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 4
    elif args.data_set == "VTAB_dSpr_Loc":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "dsprites_loc")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 16
    elif args.data_set == "VTAB_dSpr_Ori":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "dsprites_ori")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 16
    elif args.data_set == "VTAB_sNORB_Azim":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "smallnorb_azi")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 18
    elif args.data_set == "VTAB_sNORB_Ele":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "smallnorb_ele")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 9

    return dataset, nb_classes

def build_tsne_dataset(is_train, args):
    transform = build_transform(False, args)

    if is_train == True:
        split = 'train'
    elif is_train == False:
        split = "test"

    # 指定你希望数据集被下载和存储的目录
    desired_download_root = '/data/dataset/yaoyifeng/'
    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(root=desired_download_root, train=is_train, transform=transform, download=True)
        nb_classes = 100
    if args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(root=desired_download_root, train=is_train, transform=transform, download=True)
        nb_classes = 10
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == "CUB200":
        dataset = CUB200Dataset(args.data_path, split=split, transform=transform)
        nb_classes = dataset.get_class_num()
    elif args.data_set == "NABIRDS":
        dataset = NabirdsDataset(args.data_path, split=split, transform=transform)
        nb_classes = dataset.get_class_num()
    elif args.data_set == "DOGS":
        dataset = DogsDataset(args.data_path, split=split, transform=transform)
        nb_classes = dataset.get_class_num()
    elif args.data_set == "FLOWERS":
        dataset = FlowersDataset(args.data_path, split=split, transform=transform)
        nb_classes = dataset.get_class_num()
    elif args.data_set == "CARS":
        dataset = CarsDataset(args.data_path, split=split, transform=transform)
        nb_classes = dataset.get_class_num()
    elif args.data_set == "DTD":
        dataset = DTD(data_path=args.data_path, split=split, transform=transform)
        nb_classes = dataset.get_class_num()
    elif args.data_set == "GTSRB":
        dataset = GTSRB(data_path=args.data_path, split=split, transform=transform)
        nb_classes = dataset.get_class_num()
    elif args.data_set == "FOOD":
        dataset = Food101(data_path=args.data_path, split=split, transform=transform)
        nb_classes = dataset.get_class_num()
    elif args.data_set == "SVHN":
        dataset = SVHN(data_path=args.data_path, split=split, transform=transform)
        nb_classes = dataset.get_class_num()

    elif args.data_set == "VTAB_CIFAR100":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "cifar")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == "VTAB_Caltech101":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "caltech101")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 102
    elif args.data_set == "VTAB_DTD":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "dtd")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 47
    elif args.data_set == "VTAB_Flowers102":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "oxford_flowers102")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 102
    elif args.data_set == "VTAB_Pets":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "oxford_iiit_pet")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 37
    elif args.data_set == "VTAB_SVNH":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "svhn")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 10
    elif args.data_set == "VTAB_Sun397":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "sun397")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 397
    elif args.data_set == "VTAB_Camelyon":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "patch_camelyon")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 2
    elif args.data_set == "VTAB_EuroSAT":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "eurosat")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 10
    elif args.data_set == "VTAB_Resisc45":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "resisc45")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 45
    elif args.data_set == "VTAB_Retinopathy":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "diabetic_retinopathy")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 5
    elif args.data_set == "VTAB_Clevr_Count":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "clevr_count")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 8
    elif args.data_set == "VTAB_Clevr_Dist":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "clevr_dist")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 6
    elif args.data_set == "VTAB_DMLab":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "dmlab")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 6
    elif args.data_set == "VTAB_KITTI_Dist":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "kitti")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 4
    elif args.data_set == "VTAB_dSpr_Loc":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "dsprites_loc")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 16
    elif args.data_set == "VTAB_dSpr_Ori":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "dsprites_ori")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 16
    elif args.data_set == "VTAB_sNORB_Azim":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "smallnorb_azi")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 18
    elif args.data_set == "VTAB_sNORB_Ele":
        data_path = os.path.join(args.data_path, "VTAB", "vtab-1k", "smallnorb_ele")
        dataset = VTAB(root=data_path, train=is_train, transform=transform)
        nb_classes = 9

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

def build_transform_vis(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    # t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
