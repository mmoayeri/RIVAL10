from finetuner import FineTuner
from utils import *
from torchvision.utils import make_grid
from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import average_precision_score

def vit_reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def clip_vit_reshape_transform(tensor, height=14, width=14):
    result = tensor[1 :  , :].reshape(tensor.size(1),
        height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2).float()
    return result

def gradcam(model, pos, target_layer, target_category, cam, vit=False):
    if cam is None:
        model.logit_only = True
        cam = GradCAM(model=model, target_layer=target_layer, use_cuda=True)
    pos = pos.cuda()
    grayscale_cam = cam(input_tensor=pos, target_category=target_category, eigen_smooth=False)
    return grayscale_cam

def compute_saliency_thresholds(gcam, thresholds=[0.25, 0.5, 0.75]):
    ''' computes pixel saliency val per threshold s.t. sum of sals < val = t * total_sal '''
    sals = np.sort(gcam.flatten())
    tot = np.sum(sals)
    if tot == 0:
        return [1]*len(thresholds)
    running_sum = 0
    milestones = [x*tot for x in thresholds]
    curr_idx = 0
    sal_threshs = []
    i = 0
    while curr_idx < len(milestones):
        sal = sals[i]; i += 1
        running_sum += sal
        if running_sum > milestones[curr_idx]:
            sal_threshs.append(sal)
            curr_idx += 1
    return sal_threshs

def compute_mask_coverage(mask, gcam):
    threshs = compute_saliency_thresholds(gcam)
    scores = []
    for thresh in threshs:
        binary_gcam = gcam.copy()
        binary_gcam[binary_gcam < thresh] = 0
        binary_gcam[binary_gcam > 0] = 1
        scores.append(np.sum(binary_gcam * mask) / np.sum(mask))
    return scores

def gradcam_vis_and_score(model, pos, masks, target_layer, target_category, vis=True, cam=None):
    ''' For one batch, visualizes gradcams and returns IOU score '''
    grayscale_cam = []
    if type(target_category) is not list:
        target_category = [target_category] * pos.shape[0]
    for i in range(0,pos.shape[0]):
        grayscale_cam.append(gradcam(model, pos[i:i+1], target_layer, target_category[i], cam))
    grayscale_cam = np.concatenate(grayscale_cam)

    visualizations = []
    masks = masks.numpy()
    if vis:
        for i in range(pos.shape[0]):
            ind_grayscale_cam = grayscale_cam[i, :]
            rgb_img = pos[i,:].detach().cpu().numpy().swapaxes(0,1).swapaxes(1,2)
            visualization = show_cam_on_image(rgb_img, ind_grayscale_cam)
            visualizations.append(torch.tensor(visualization))
        visualizations = torch.stack(visualizations, axis=-1).permute([3,2,0,1])
        grid = toPIL(make_grid(visualizations, nrow=5))
        r, g, b = grid.split()
        grid = Image.merge('RGB', (b, g, r))
    else:
        grid = None

    masks = masks[:,0]
    masks[masks > 0] = 1
    N = masks.shape[0]

    grayscale_cam[np.isnan(grayscale_cam)] = 0
    ious = [intersection_over_union_and_dice([masks[i], grayscale_cam[i]])[0] for i in range(N)]
    fracs_inside = [np.sum(grayscale_cam[i] * masks[i]) / np.sum(grayscale_cam[i]) for i in range(N)]
    mask_coverages = [compute_mask_coverage(masks[i], grayscale_cam[i]) for i in range(N)]
    delta_densities = [delta_saliency_density(grayscale_cam[i], masks[i]) for i in range(N)]
    aps = [average_precision_score(masks[i].flatten(), grayscale_cam[i].flatten()) for i in range(N)]

    return grid, fracs_inside, mask_coverages, aps, delta_densities, ious

def obtain_samples_from_class(loader, class_idx, num_samples=36):
    all_class_imgs, all_class_masks = [], []
    ctr = 0
    for imgs, masks, labels in loader:
        if ctr > num_samples:
            break
        class_imgs = imgs[labels==class_idx]
        class_masks = masks[labels==class_idx]
        ctr += class_imgs.shape[0]
        all_class_imgs.append(class_imgs)
        all_class_masks.append(class_masks)
    imgs = torch.cat(all_class_imgs, axis=0)
    masks = torch.cat(all_class_masks, axis=0)
    imgs, masks = [x[:num_samples] for x in [imgs, masks]]
    return imgs, masks

def get_cam_obj(model, target_layer, mtype, camtype='gradcam'):
    if 'clip_ViT' in mtype:
        if '32' in mtype:
            reshape_transform = lambda x : clip_vit_reshape_transform(x, height=7, width=7)
        else:
            reshape_transform = lambda x : clip_vit_reshape_transform(x, height=14, width=14)
    elif 'clip_RN' in mtype:
        reshape_transform = lambda x : x.float()
    elif 'deit' in mtype or 'vit' in mtype:
        reshape_transform = vit_reshape_transform
    else:
        reshape_transform = lambda x : x
    
    if camtype == 'gradcam':
        cam = GradCAM(model=model, target_layer=target_layer, use_cuda=True, reshape_transform=reshape_transform)
    elif camtype == 'scorecam':
        cam = ScoreCAM(model=model, target_layer=target_layer, use_cuda=True, reshape_transform=reshape_transform)

    return cam

def compute_all_bg_fg_scores(mtype, loader, camtype='gradcam'):
    '''
    Obtains saliency maps for all inputs and computes foreground alignment using five metrics.
    Caches results.
    '''
    model, target_layer = obtain_model(mtype)
    cam = get_cam_obj(model, target_layer, mtype)

    stats = load_cached_results(save_root='./bg_fg/by_preds/{}'.format(camtype if camtype != 'gradcam' else ''), 
            key='{}_bg_fg_scores_by_preds'.format(mtype))

    all_fracs_inside = dict({target:[] for target in range(10)})
    all_logits = dict({target:[] for target in range(10)})
    all_labels = dict({target:[] for target in range(10)})
    all_mask_coverages = dict({target:[] for target in range(10)})
    all_aps = dict({target:[] for target in range(10)})
    all_delta_densities = dict({target:[] for target in range(10)})
    all_ious = dict({target:[] for target in range(10)})
    worst_fracs_inside = dict({target:[(1,0)]*5 for target in range(10)})
    best_fracs_inside = dict({target:[(0,0)]*5 for target in range(10)})
    worst_aps = dict({target:[(1,0)]*5 for target in range(10)})
    best_aps = dict({target:[(0,0)]*5 for target in range(10)})

    for imgs, masks, labels in tqdm(loader):
        # obtain model predictions
        imgs = imgs.cuda()
        logits = model(imgs)
        preds = logits.argmax(1)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        for target in range(10):
            subset = imgs[preds == target]
            subset_masks = masks[preds == target]
            subset_logits = logits[preds == target].detach().cpu().numpy()
            subset_labels = labels[preds == target].detach().cpu().numpy()

            if subset.shape[0] == 0:
                continue
            grid, fracs_inside, mask_coverages, aps, delta_densities, ious = gradcam_vis_and_score(model, subset, subset_masks, 
                                                                             target_layer, target, vis=False, cam=cam)
            
            all_fracs_inside[target].extend(fracs_inside)
            all_mask_coverages[target].extend(mask_coverages)
            all_aps[target].extend(aps)
            all_delta_densities[target].extend(delta_densities)
            all_ious[target].extend(ious)
            all_logits[target].extend(subset_logits)
            all_labels[target].extend(subset_labels)

    dcr_idx_to_class_dict = get_dcr_idx_to_class_dict()
    results_dict = dict({'fracs_inside': all_fracs_inside, 
                         'mask_coverages': all_mask_coverages,
                         'aps': all_aps,
                         'ious': all_ious,
                         'delta_densities': all_delta_densities,
                         'all_logits': all_logits,
                         'all_labels': all_labels,
                         })
    for k in results_dict:
        stats[k] = results_dict[k]
    if camtype != 'gradcam':
        cache_results(stats, save_root='./bg_fg/by_preds/{}'.format(camtype), key='{}_bg_fg_scores_by_preds'.format(mtype))
    else:
        cache_results(stats, save_root='./bg_fg/by_preds/', key='{}_bg_fg_scores_by_preds'.format(mtype))

    for i in range(10):
        print(dcr_idx_to_class_dict[i])
        print(np.nanmean(all_fracs_inside[i]), np.nanstd(all_fracs_inside[i]))
        print([(np.nanmean(all_mask_coverages[i], axis=0)[j], np.nanstd(all_mask_coverages[i], axis=0)[j]) for j in range(3)])
        print([x[0] for x in worst_fracs_inside[i]])

if __name__ == '__main__':
    testset = RIVAL10(train=False, return_masks=True)
    loader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False)

    _ALL_MTYPES = ['resnet18', 'resnet50', 'resnet101','resnet152', 
            'robust_resnet18', 'robust_resnet50', 'vit_tiny', 'vit_small', 'vit_base', 
            'deit_tiny', 'deit_small', 'deit_base', 'simclr', 'clip_RN50', 'clip_RN101', 
            'clip_ViT-B16', 'clip_ViT-B32']
    for mtype in _ALL_MTYPES:
        compute_all_bg_fg_scores(mtype, loader)
