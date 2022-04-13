import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
import os
import glob
import json
import random
import matplotlib.image as mpimg
from PIL import Image
from pathlib import Path

# REDACTED FOR ANONYMITY
_ATTR_DATA_ROOT = None
_IMAGENET_ROOT = None
_S3_IMAGENET_ROOT = None
_LABEL_MAPPINGS = './datasets/label_mappings.json'
_WNID_TO_CLASS = '/datasets/wnid_to_class.json'
_NO_EO_MASKS_PATH = None
_MASKS_TEMPLATE_ROOT = None

def resize(img): 
    return np.array(Image.fromarray(np.uint8(img)).resize((224,224))) / 255

def to_3d(img):
    return np.stack([img, img, img], axis=-1)

def save_uri_as_img(uri, fpath='tmp.png'):
    ''' saves raw mask and returns it as an image'''
    binary_data = a2b_base64(uri)
    with open(fpath, 'wb') as f:
        f.write(binary_data)
    img = mpimg.imread(fpath)
    img = resize(img)
    # binarize mask
    img = np.sum(img, axis=-1)
    img[img != 0] = 1
    img = to_3d(img)
    return img

class AttrDataset(Dataset):
    def __init__(self, attr, train=True, return_label_and_eo_mask=False):
        # self.transform = transform
        self.attr = attr
        self.return_label_and_eo_mask = return_label_and_eo_mask
        self.train= train
        self.split = 'train' if train else 'test'
        self.data_root = _ATTR_DATA_ROOT if self.train else _ATTR_DATA_ROOT+'test/'
        self.eo_masks_root = _MASKS_TEMPLATE_ROOT.format(self.split)
        with open(_LABEL_MAPPINGS, 'r') as f:
            self.label_mappings = json.load(f)
        with open(_WNID_TO_CLASS, 'r') as f:
            self.wnid_to_class = json.load(f)
        with open(_NO_EO_MASKS_PATH, 'r') as f:
            self.failed_urls = json.load(f)

        self.init_instances_lists()

        self.subset_idx = np.arange(len(self.positive_instances))

        self.filter_instances_using_subset_idx()

        self.train = train
        self.resize = transforms.Resize((224,224))
        self.num_instances = len(self.subset_idx)
        print('{}ing dataset for attribute {} loaded, with {} samples.'
              .format('Train' if self.train else 'test', self.attr, self.num_instances))

    def __len__(self):
        return self.num_instances

    def recover_label(self, og_path):
        wnid = og_path.split('/')[-1].split('_')[0]
        label = self.label_mappings[self.wnid_to_class[wnid]][1]
        return label

    def __getitem__(self, index):
        og_path = self.positive_instances[index]
        # print(og_path)
        attr_mask_path = os.path.join(self.data_root, self.attr, 'masks', og_path.split('/')[-1])
        eo_mask_path = os.path.join(self.eo_masks_root, og_path.split('/')[-1])
        # randomly sample negative image
        # neg_og_path = self.negative_instances[np.random.randint(self.num_instances)]
        # deterministically sample negative image
        neg_og_path = self.negative_instances[index]
        og, mask, eo_mask, neg_og = [self.resize(Image.open(f)) for f in [og_path, attr_mask_path, eo_mask_path, neg_og_path]]
        og, mask, eo_mask, neg_og = [np.array(img) for img in [og, mask, eo_mask, neg_og]]
        if len(og.shape) == 2:
            og = np.stack([og, og, og], axis=-1)
        if len(neg_og.shape) == 2:
            neg_og = np.stack([neg_og, neg_og, neg_og], axis=-1)
        if len(mask.shape) == 2:
            mask = np.stack([mask, mask, mask], axis=-1)
        if len(eo_mask.shape) == 2:
            eo_mask = np.stack([eo_mask, eo_mask, eo_mask], axis=-1)


        imgs = [Image.fromarray(img) for img in [og, mask, eo_mask, neg_og]]
        og, mask, eo_mask, neg_og = self.transform(imgs)
        mask[mask != 0] = 1; eo_mask[mask != 0] = 1
        if self.return_label_and_eo_mask:
            label = self.recover_label(og_path)
            return og, neg_og, mask, eo_mask, label
        else:
            return og, neg_og, mask

    def transform(self, imgs):
        transformed_imgs = []
        i, j, h, w = transforms.RandomResizedCrop.get_params(imgs[0], scale=(0.8,1.0),ratio=(0.75,1.25))
        coin_flip = (random.random() < 0.5)
        for ind, img in enumerate(imgs):
            if self.train:
                img = TF.crop(img, i, j, h, w)

                if coin_flip:
                    img = TF.hflip(img)

            img = TF.to_tensor(self.resize(img))
            
            if img.shape[0] == 1:
                img = torch.cat([img, img, img], axis=0)
            
            transformed_imgs.append(img)

        return transformed_imgs

    def has_eo_mask(self, fpath):
        wnid_and_fname = fpath.split('/')[-1]
        first_underscore_ind = wnid_and_fname.find('_')
        wnid = wnid_and_fname[:first_underscore_ind]
        fname = wnid_and_fname[first_underscore_ind+1:]
        ext = '/'.join([self.split, wnid, fname])
        imagenet_path = _IMAGENET_ROOT + ext
        no_eo_mask = (imagenet_path in self.failed_urls)
        # if not no_eo_mask:
        #     try:
        #         mask_path = os.path.join(self.eo_masks_root, fpath.split('/')[-1])
        #         mask = Image.open(mask_path)
        #     except:
        #         print('Failed on {}. Added to failed urls'.format(imagenet_path))
        #         self.failed_urls.append(imagenet_path)
        return not no_eo_mask

    def init_instances_lists(self):
        all_instances = {}
        ctr = 0
        positive_instances = []
        for file in glob.glob(self.data_root+self.attr+'/original/*'):
            if '.json' not in file:
                if self.has_eo_mask(file):
                    positive_instances.append(file)

        
        negative_instances = []
        for file in glob.glob(self.data_root+self.attr+'/superimposed/*'):
            if '.json' in file:
                with open(file, 'r') as f:
                    label_dict = json.load(f)
                bg_url = label_dict['bg_url']
                neg_og_url = _IMAGENET_ROOT + bg_url.split('imagenet/')[-1]
                negative_instances.append(neg_og_url)

        self.positive_instances = positive_instances
        self.negative_instances = negative_instances

    def filter_instances_using_subset_idx(self):
        positive_instances_to_keep = []
        negative_instances_to_keep = []

        for ind in self.subset_idx:
            positive_instances_to_keep.append(self.positive_instances[ind])
            negative_instances_to_keep.append(self.negative_instances[ind])
        
        self.positive_instances = positive_instances_to_keep
        self.negative_instances = negative_instances_to_keep
