import matplotlib.pyplot as plt
from utils import load_cached_results, get_dcr_idx_to_class_dict
from bg_fg_saliency import obtain_model, get_cam_obj
from pytorch_grad_cam.utils.image import show_cam_on_image
import pickle
import numpy as np
import torch
from datasets.local_rival10 import *
from datasets import *
from tqdm import tqdm
from torchvision.utils import make_grid

dcr_idx_to_class_dict = get_dcr_idx_to_class_dict()
dcr_class_to_idx = dict({dcr_idx_to_class_dict[k]:k for k in dcr_idx_to_class_dict})

def get_name(mtype):
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

def view_single_img(img_tens, mtype, ax, true_class_name):
    model, target_layer = obtain_model(mtype)
    cam = get_cam_obj(model, target_layer, mtype)
    img_tens = img_tens.unsqueeze(0).cuda()
    logits = model(img_tens)
    probs = torch.softmax(logits, 1)
    pred_prob, pred_class = torch.max(probs, 1)

    grayscale_cam = cam(input_tensor=img_tens, target_category=pred_class, eigen_smooth=False)
    rgb_img = img_tens[0].detach().cpu().numpy().swapaxes(0,1).swapaxes(1,2)
    visualization = show_cam_on_image(rgb_img, grayscale_cam[0])[:,:,::-1]
    # img_np = img_tens.squeeze(0).detach().cpu().numpy().swapaxes(0,1).swapaxes(1,2)
    # pred_class = 0
    # img_np = np.ones((224,224,3))
    ax.imshow(visualization)
    pred_class_name = dcr_idx_to_class_dict[pred_class.detach().item()]
    ax.set_xticks([])
    ax.set_yticks([])
    c = 'red' if pred_class_name != true_class_name else 'black'
    print(pred_class_name, true_class_name)
    ax.text(0, 240, '{} predicts:'.format(get_name(mtype)), size=12)
    ax.text(0, 260, pred_class_name.capitalize(), color=c, size=12)
    ax.text(40, 260, 'with probability: {:.1f}%'.format(100*pred_prob.item()), color=c, size=12)

    
def get_img(mtype, metric_key, pred_class_name, ind, worst=True):
    with open('./bg_fg/by_preds/{}_bg_fg_scores_by_preds.pkl'.format(mtype), 'rb') as fp:
        scores = pickle.load(fp)
    i = dcr_class_to_idx[pred_class_name]
    l = scores['{}_{}'.format('worst' if worst else 'best', metric_key)][i]
    img = l[ind][1]
    return img

mistakes = [
    ('simclr', 'ship', 'ship', 'delta_densities', 1),
    ('simclr', 'ship', 'bird', 'delta_densities', 3),
    ('robust_resnet18', 'ship', 'ship', 'delta_densities', 4),
    ('robust_resnet18', 'ship', 'deer', 'delta_densities', 3),
    ('resnet18', 'bird', 'bird', 'delta_densities', 0),
    ('resnet18', 'bird', 'frog', 'delta_densities', 4),
]

other_instances = [
    ('deit_tiny', 'bird', 'bird', 'delta_densities', 2),
    ('deit_tiny', 'bird', 'bird', 'delta_densities', 1),
    ('deit_base', 'bird', 'bird', 'mask_coverages_25', 1),
    ('deit_small', 'bird', 'bird', 'mask_coverages_25', 3),
    ('vit_tiny', 'ship', 'ship', 'delta_densities', 2),
    ('vit_tiny', 'ship', 'ship', 'delta_densities', 3),
]


plot_keys = [
    ('simclr', 'ship', 'ship', 'delta_densities', 1),
    ('simclr', 'ship', 'bird', 'delta_densities', 3),
#     ('resnet18', 'ship', 'ship', 'delta_densities', 1),
#     ('robust_resnet18', 'ship', 'deer', 'delta_densities', 3),
    ('robust_resnet18', 'ship', 'ship', 'delta_densities', 4),
    ('robust_resnet18', 'ship', 'plane', 'delta_densities', 0),
    ('resnet152', 'ship', 'ship', 'delta_densities', 2),
    ('resnet101', 'ship', 'plane', 'delta_densities', 1),

    ('simclr', 'bird', 'bird', 'delta_densities', 1),
    ('simclr', 'bird', 'deer', 'delta_densities', 0),
    ('robust_resnet50', 'bird', 'bird', 'delta_densities', 1),
    ('robust_resnet50', 'bird', 'frog', 'delta_densities', 4),
    ('deit_tiny', 'bird', 'bird', 'delta_densities', 2),
    ('deit_tiny', 'bird', 'bird', 'delta_densities', 1),

]

def view_spurious(mistakes, other_instances):
    for keys, title in zip([mistakes, other_instances], ['mistakes','other']):
        n_col = len(keys) // 2
        f, axs = plt.subplots(2,n_col, figsize=(3*n_col,6))
        for i in range(n_col):
            top_mtype, top_pred_class_name, top_true_class, top_metric_key, top_ind = keys[2*i] 
            bot_mtype, bot_pred_class_name, bot_true_class, bot_metric_key, bot_ind = keys[2*i+1] 

            top_img, bot_img = [get_img(*args) for args in [(top_mtype, top_metric_key, top_pred_class_name, top_ind),
                                                            (bot_mtype, bot_metric_key, bot_pred_class_name, bot_ind)]]
            view_single_img(top_img, top_mtype, axs[0,i], top_true_class)
            view_single_img(bot_img, bot_mtype, axs[1,i], bot_true_class)

        f.tight_layout()
        f.savefig('./visualizations/{}.png'.format(title), dpi=300)


def view_rival10_examples(classes_to_skip=[]):
    dset = LocalRIVAL10(train=True, masks_dict=True, no_aug=True, keep_clean=True)
    loader = torch.utils.data.DataLoader(dset, batch_size=50, shuffle=True)
    examples_per_class = [0] * 10
    for c in classes_to_skip:
        examples_per_class[c] = 1
    for out in loader:
        imgs, class_names, class_labels, attr_labels, attr_masks = [out[x] 
            for x in ['img', 'og_class_name', 'og_class_label', 'attr_labels', 'attr_masks']]
        for i in range(imgs.shape[0]):
            class_label = class_labels[i].item()
            if (examples_per_class[class_label] > 0) or (torch.sum(attr_labels[i]) < 2):
                continue
            else:
                examples_per_class[class_label] += 1

            on_attrs = np.where(attr_labels[i] == 1)[0]
            print(class_names[i])
            print([idx_to_attr(x) for x in on_attrs])

            f, axs = plt.subplots(1,4, figsize=(9,2.5))
            axs[0].imshow(imgs[i].numpy().swapaxes(0,1).swapaxes(1,2))
            axs[0].set_title('Class: {}'.format(class_names[i].capitalize()))
            axs[0].set_xticks([]); axs[0].set_yticks([])

            object_only = imgs[i] * attr_masks[i][-1]
            axs[1].imshow(object_only.numpy().swapaxes(0,1).swapaxes(1,2))
            axs[1].set_title('Object')
            axs[1].set_xticks([]); axs[1].set_yticks([])
            for j in range(2,4):
                axs[j].set_xticks([]); axs[j].set_yticks([])
                attr_only = imgs[i] * attr_masks[i][on_attrs[j-2]]
                axs[j].imshow(attr_only.numpy().swapaxes(0,1).swapaxes(1,2))
                axs[j].set_title(idx_to_attr(on_attrs[j-2]).capitalize())
            # axs[2].imshow(attr_masks[i][on_attrs[2]].numpy().swapaxes(0,1).swapaxes(1,2))
            # axs[2].set_title(idx_to_attr(on_attrs[2]))
            f.tight_layout()
            f.savefig('./visualizations/examples/{}.png'.format(class_names[i]))
            # f.close()
        stop = 1
        for x in examples_per_class:
            stop = x * stop
        if stop != 0:
            break

def view_attr_swapped_imgs(attr='ears'):
    dset = AttrDataset(attr, train=False)
    loader = torch.utils.data.DataLoader(dset, batch_size=3, shuffle=False)
    pos, neg, masks = next(iter(loader))
    superimposed = pos * masks + neg * (1-masks)
    removed = pos * (1-masks) + neg * masks
    g1, g2 = [make_grid(x, nrow=3).numpy().swapaxes(0,1).swapaxes(1,2) for x in [removed, superimposed]]
    f, axs = plt.subplots(2,1,figsize=(5,4.16))
    for g, ax in zip([g1,g2], axs):
        ax.imshow(g)
        ax.set_xticks([])
        ax.set_yticks([])
    axs[0].set_title('{} Removed'.format(attr.capitalize()))
    axs[1].set_title('{} Superimposed'.format(attr.capitalize()))
    f.tight_layout()
    f.savefig('./visualizations/attr_swapped_{}.png'.format(attr), dpi=300) 


### To generate attr_swapped examples
view_attr_swapped_imgs()

### To generate dataset samples
# class_names_to_skip = []#['bird', 'equine', 'dog', 'truck', 'cat', 'ship', 'deer', 'plane', 'car']
# dcr_idx_to_class_dict = get_dcr_idx_to_class_dict()
# dcr_class_to_idx_dict = dict({dcr_idx_to_class_dict[c]:c for c in dcr_idx_to_class_dict})
# classes_to_skip = [dcr_class_to_idx_dict[cn] for cn in class_names_to_skip]
# view_rival10_examples(classes_to_skip)

#     ('deit_tiny', 'ship', 'ship', 'delta_densities', 4),
'''
All of vit_tiny_worst_delta_densities ships show water!
deit_tiny_worst_delta_densities: bird 0, 1, 2,

deit_base_worst_mask_coverages_25: bird 1 <-- bird feeder
deit_small_worst_mask_coveragegs_25: bird 3 <-- bird feeder

resnet50_worst_delta_densities: bird 0
simclr_worst_delta_densities: bird 0,1 <-- misclassified deer; ship 0,1,3,4 <-- misclassified dogs and bird

vit_base_worst_delta_densities: bird 1,2 <-- bird feeders
robust_resnet50_worst_delta_densities: bird 3,4 <-- twigs, misclassified frog
robust_resnet18_worst_fracs_inside: bird 2 <-- misclassified truck
'''