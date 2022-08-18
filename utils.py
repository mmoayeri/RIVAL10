import torch
import numpy as np
import json
import pickle
import os
from datasets import *
from datasets.rival10 import _LABEL_MAPPINGS
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from finetuner import *

def get_model_name(mtype):
    out = mtype
    out = out.replace('resnet', 'ResNet')
    out = out.replace('RN', 'ResNet')
    out = out.replace('clip', 'CLIP')
    out = out.replace('vit', 'ViT')
    out = out.replace('robust', 'Robust')
    out = out.replace('deit', 'DeiT')
    out = out.replace('simclr', 'SimCLR')
    out = out.replace('small', '(Small)')
    out = out.replace('base', '(Base)')
    out = out.replace('tiny', '(Tiny)')
    out.replace('B', 'B//')
    out = ' '.join(str(out).split('_'))
    return out

def binarize(m):
    m = np.array(m)
    m[np.isnan(m)] = 0
    m = m / np.max(m) if np.max(m) != 0 else m
    m[m>0.5] = 1
    m[m<0.5] = 0
    return m

def intersection_over_union_and_dice(masks, eps=1e-10):
    masks = [binarize(m) for m in masks]
    intersection = masks[0]
    for mask in masks[1:]:
        intersection = intersection * mask
    intersection = np.sum(intersection > 0)
    union = np.sum(np.sum(masks, axis=0) > 0)
    dice = (2*intersection + eps) / (union + intersection + eps) #if union != 0 else NaN
    iou = (intersection + eps) / (union + eps)
    return iou, dice

def delta_saliency_density(gcam, mask):
    density_fg = np.sum(gcam * mask) / np.sum(mask)
    density_bg = np.sum(gcam * (1-mask)) / np.sum(1-mask)
    return density_fg - density_bg

def l2_normalize(tens):
    # normalizes tensor along batch dimension
    norms = torch.norm(tens.view(tens.shape[0], -1), p=2, dim=1)
    factor = torch.ones(tens.shape).cuda()
    factor = factor / norms.view(-1,1,1,1)
    normalized = tens * factor
    return normalized

### generalization heatmaps
def heatmap(accs, xlabels, ylabels, fig_save_root, title, fsr_is_dir=True):
    plt.clf()
    plt.figure(figsize=(3+1.5*len(xlabels),3+1.5*len(ylabels)), dpi=300)
    fig, ax = plt.subplots()
    im = ax.imshow(accs)
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    thresh = np.max(accs) * 0.75
    get_color = lambda x: "black" if x > thresh else "w"
    # Loop over data dimensions and create text annotations.
    for i in range(len(ylabels)):
        for j in range(len(xlabels)):
            text = ax.text(j, i, np.round(accs[i, j],2), size=7,
                        ha="center", va="center", color=get_color(accs[i,j]))

    # ax.set_ylabel("Trained on")
    # ax.set_xlabel("Accuracy on")
    ax.set_title(title)
    plt.tight_layout()
    fpath = '{}/heatmap.png' if fsr_is_dir else '{}_heatmap.png'
    plt.savefig(fpath.format(fig_save_root), bbox_inches='tight')
    plt.close()

### Loading/Caching Results

def load_cached_results(save_root, key, overwrite=False):
    save_path = save_root + '/{}.pkl'.format(key)
    already_exists = os.path.exists(save_path)
    print('\nChecking {} for {} results dict'.format(save_path, key))
    if overwrite or (not already_exists):
        results_dict = dict({})
        print('\nNo cached results used {}'.format('because of overwrite flag. \
                STOP RUNNING IF OVERWRITE IS NOT INTENDED.' if overwrite else ''))
    else:
        with open(save_path, 'rb') as f:
            results_dict = pickle.load(f)
        print('\nLoaded cached results.')
    return results_dict

def cache_results(results_dict, save_root, key):
    save_path = save_root + '/{}.pkl'.format(key)
    with open(save_path, 'wb') as f:
        pickle.dump(results_dict, f)
    print('Cached {} results to {}'.format(key, save_path))

def obtain_model(mtype, fg_only=False):
    finetuner = FineTuner(mtype=mtype, fix_ftrs=True, fg_only=fg_only)
    finetuner.restore_model()
    model = finetuner.model.cuda()
    for param in model.feat_net.parameters():
        param.requires_grad = True
    target_layer = finetuner.gradcam_layer
    return model, target_layer

### Miscellaneous

def get_dcr_idx_to_class_dict():
    with open(_LABEL_MAPPINGS, 'r') as f:
        label_mappings = json.load(f)
    dcr_idx_to_class_dict = dict()
    for classname, ind in label_mappings.values():
        dcr_idx_to_class_dict[ind] = classname
    return dcr_idx_to_class_dict