import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

import glob
import json
import matplotlib.image as mpimg
from PIL import Image
import pickle
from tqdm import tqdm

from .constants import *

from typing import Optional

def attr_to_idx(attr):
    return RIVAL10_constants._ALL_ATTRS.index(attr)

def idx_to_attr(idx):
    return RIVAL10_constants._ALL_ATTRS[idx]

def resize(img): 
    return np.array(Image.fromarray(np.uint8(img)).resize((224,224))) / 255

def to_3d(img):
    return np.stack([img, img, img], axis=-1)

def decode_base64_as_img(uri):
    ''' encode base64 as mask '''
    import base64
    from io import BytesIO
    image_data = base64.b64decode(uri)
    image_data = BytesIO(image_data)
    img = mpimg.imread(image_data, format='jpg')

    img = resize(img)
    # binarize mask
    img = np.sum(img, axis=-1)
    img[img != 0] = 1
    img = to_3d(img)
    return img

class LocalRIVAL10(Dataset):
    def __init__(self, train:bool=True, 
                 cherrypick_list:list[str]=None,
                 masks_dict:bool=True, 
                 transform:Optional[transforms.Compose]=None,
                 verbose:str="key"):
        '''
        Set masks_dict to be true to include tensor of attribute segmentations when retrieving items.
        Use cherrypick_list to fit different shapes of output, e.g. set cherrypick_list to be ["img", "og_class_label"] -> img, label = rival10.__getitem__(0)

        See __getitem__ for more documentation. 
        '''
        self.train = train
        self.data_root = RIVAL10_constants._RIVAL10_DIR.format('train' if self.train else 'test')
        self.cherrypick_list = cherrypick_list
        # self.spss_output = spss_output
        self.masks_dict = masks_dict
        self.transform = transform
        self.verbose = verbose
        # quiet, key, full

        self.instance_types = ['ordinary']
        # NOTE: 
        # if include_aug:
        #     self.instance_types += ['superimposed', 'removed']
        
        self.instances = self.collect_instances()

        with open(RIVAL10_constants._LABEL_MAPPINGS, 'r') as f:
            self.label_mappings = json.load(f)
        with open(RIVAL10_constants._WNID_TO_CLASS, 'r') as f:
            self.wnid_to_class = json.load(f)

    def get_rival10_og_class(self, img_url):
        wnid = img_url.split('/')[-1].split('_')[0]
        inet_class_name = self.wnid_to_class[wnid]
        classname, class_label = self.label_mappings[inet_class_name]
        return classname, class_label

    def collect_instances(self):
        self.instances_by_type = dict()
        self.all_instances = []
        for subdir in self.instance_types:
            instances = []
            dir_path = self.data_root + subdir
            itr = glob.glob(dir_path+'/*')
            if self.verbose == "full" or self.verbose == "key":
                itr = tqdm(glob.glob(dir_path+'/*'))
            for f in itr:
                if '.JPEG' in f and 'merged_mask' not in f:
                    img_url = f
                    label_path = f[:-5] + '_attr_labels.npy'
                    merged_mask_path = f[:-5] + '_merged_mask.JPEG'
                    mask_dict_path = f[:-5] + '_attr_dict.pkl'
                    instances.append((img_url, label_path, merged_mask_path, mask_dict_path))
            self.instances_by_type[subdir] = instances.copy()
            self.all_instances.extend(self.instances_by_type[subdir])

    def __len__(self):
        return len(self.all_instances)

    def merge_all_masks(self, mask_dict):
        merged_mask = np.zeros((224,224,3))
        for attr in mask_dict:
            if attr == 'entire-object':
                continue
            mask_uri = mask_dict[attr]
            mask = decode_base64_as_img(mask_uri)
            merged_mask = mask if merged_mask is None else mask + merged_mask
        merged_mask[merged_mask > 0] = 1
        return merged_mask

    def __getitem__(self, i):
        '''
        Returns dict with following keys:
            img
            attr_labels: binary vec with 1 for present attrs
            changed_attr_labels: binary vec with 1 for attrs that were removed or pasted (not natural)
            merged_mask: binary mask with 1 for any attribute region
            attr_masks: tensor w/ mask per attribute. Masks are empty for non present attrs
        '''
        img_url, label_path,  merged_mask_path, mask_dict_path = self.all_instances[i]
        if self.verbose == "full":
            print("Now loading:", img_url, label_path,  merged_mask_path, mask_dict_path)
        # get rival10 info for original image (label may not hold for attr-augmented images)
        class_name, class_label = self.get_rival10_og_class(img_url)

        # load img
        img = Image.open(img_url)
        if img.mode == 'L':
            img = img.convert("RGB")

        # load labels
        labels = np.load(label_path)
        attr_labels = torch.Tensor(labels[0]).long()
        changed_attrs = torch.Tensor(labels[1]).long() # attrs that were added or removed

        merged_mask_img = Image.open(merged_mask_path)
        imgs = [self.transform(img), self.transform(merged_mask_img)]
        if self.masks_dict:
            try:
                with open(mask_dict_path, 'rb') as fp:
                    mask_dict = pickle.load(fp)
            except:
                print(f"Error occured when loading {mask_dict_path}!")
                mask_dict = dict()
            if self.verbose == "full":
                print(mask_dict)
            for attr in mask_dict:
                i
                mask_uri = mask_dict[attr]
                mask = decode_base64_as_img(mask_uri)
                imgs.append(torch.from_numpy(np.uint8(mask)).permute((2, 0, 1)))

        img = imgs.pop(0)
        merged_mask = imgs.pop(0)
        out = dict({'img':img, 
                    'attr_labels': attr_labels, 
                    'changed_attrs': changed_attrs,
                    'merged_mask' :merged_mask,
                    'og_class_name': class_name,
                    'og_class_label': class_label})

        if self.masks_dict:
            attr_masks = [torch.zeros(img.shape) for i in range(len(RIVAL10_constants._ALL_ATTRS)+1)]
            for i, attr in enumerate(mask_dict):
                # if attr == 'entire-object':
                ind = -1 if attr == 'entire-object' else attr_to_idx(attr)
                attr_masks[ind] = imgs[i]
            out['attr_masks'] = torch.stack(attr_masks)
        
        if self.cherrypick_list is not None:
            return (out[key] for key in self.cherrypick_list)

        return out