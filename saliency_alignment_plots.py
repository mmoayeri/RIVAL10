from finetuner import FineTuner
from utils import *
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from bg_fg_saliency import gradcam_vis_and_score, get_cam_obj

def get_metric_name(mkey):
    if mkey == 'ious':
        # mname = 'IOU of GradCAM \nand Object Mask'
        mname = 'IOU'
    elif mkey == 'delta_densities':
        mname = '$\Delta$ Densities'
    elif mkey == 'aps':
        mname = 'Average Precision'
    elif mkey == 'fracs_inside':
        mname = 'Saliency Precision'
    elif mkey == 'mask_coverages_25':
        mname = 'Saliency Recall'
    return mname

groups = dict({
    'DeiTs': (['deit_tiny', 'deit_small', 'deit_base'], 's'),
    'CLIP\nViTs': (['clip_ViT-B16', 'clip_ViT-B32'], 's'),
    'ViTs': (['vit_tiny', 'vit_small', 'vit_base'], 's'),
    'CLIP\nResNets': (['clip_RN50', 'clip_RN101'], '^'),
    'SimCLR': (['simclr'], '^'),
    'Robust\nResNets': (['robust_resnet18', 'robust_resnet50'], '^'),
    'ResNets': (['resnet18', 'resnet50', 'resnet101', 'resnet152'], '^')
})
cmap = cm.get_cmap('gist_ncar')
colors = [cmap(x/len(groups)) for x in range(len(groups))]
dcr_idx_to_class_dict = get_dcr_idx_to_class_dict()

def process_scores(metric_args, scores_dict):
    ''' Utility function for reading scores '''
    if 'mask_coverage' in metric_args['key']:
        all_raw = scores_dict['mask_coverages']
        out = dict({i:np.array(all_raw[i])[:,metric_args['mc_ind']] for i in range(10)})
    else:
        raw = scores_dict[metric_args['key']]
        out = dict({i:np.array(raw[i]) for i in range(10)})
    return out

def plot_bars(avgs_by_mtype, avgs_by_class, mkey, axs=None):
    ''' 
    Generates figure 9 in paper and figure 8 in supp.
    '''
    ious_by_group = []
    ious_by_class =[]
    for g in groups:
        ious = []
        for mtype in groups[g][0]:
            ious.append(avgs_by_mtype[mtype])
            ious_by_class.append(avgs_by_class[mtype])
            print(ious_by_class[-1].shape)
        ious_by_group.append(np.average(ious))
    ious_by_class = np.average(ious_by_class, 0)
    f, axs = plt.subplots(2,1, figsize=(3,5))#, sharey=True)
    x1s = np.arange(len(groups))
    x2s = np.arange(10)
    group_names = list(groups.keys())
    class_names = [dcr_idx_to_class_dict[i] for i in range(10)]
    for iou, x, c in zip(ious_by_group, x1s, colors):
        axs[0].bar(x, iou, color=c)
    cmap2 = cm.get_cmap('viridis')
    colors2 = [cmap2(x/len(class_names)) for x in range(len(class_names))]
    for iou, x, c in zip(ious_by_class, x2s, colors2):
        axs[1].bar(x, iou, color=c)
    axs[0].set_ylabel(get_metric_name(mkey))
    axs[0].yaxis.label.set_size(12)
    axs[0].set_xticks(x1s)
    axs[0].set_xticklabels(group_names, rotation='vertical')
    axs[1].set_xticks(x2s)
    axs[1].set_xticklabels(class_names, rotation='vertical')
    axs[1].set_ylabel(get_metric_name(mkey)); axs[1].yaxis.label.set_size(12)
    f.tight_layout()
    f.savefig('./bg_fg/bar_plots/{}.png'.format(mkey), dpi=300)


def gen_heatmap(metric_args, bar=False):
    '''
    Generates figure 8 in supp.
    '''
    fs = ['resnet18', 'resnet50', 'resnet101', 'resnet152', 
          'robust_resnet18', 'robust_resnet50', 'deit_tiny', 'deit_small', 'deit_base', 
          'vit_tiny', 'vit_small', 'vit_base', 
          'simclr', 'clip_RN50','clip_RN101',  'clip_ViT-B16', 'clip_ViT-B32']

    all_avgs = dict({m_args['key']:[] for m_args in metric_args})
    ylabels = []
    for f in fs:
        with open('./bg_fg/by_preds/{}_bg_fg_scores_by_preds.pkl'.format(f), 'rb') as fp:
            d = pickle.load(fp)
        for m_args in metric_args:
            scores = process_scores(m_args, d)
            avg_by_class = [np.nanmean(scores[i]) for i in range(10)]
            avg_by_class.append(np.average(avg_by_class))
            all_avgs[m_args['key']].append(avg_by_class)
        ylabels.append(f)
    dcr_idx_to_class_dict = get_dcr_idx_to_class_dict()
    xlabels = [dcr_idx_to_class_dict[i] for i in range(10)] + ['Average']
    for j, m in enumerate(all_avgs):
        avgs = np.zeros((len(fs)+1,11))
        avgs[:len(fs),:] = np.array(all_avgs[m])
        avgs[len(fs),:] = np.average(avgs[:len(fs),:], 0)
        heatmap(avgs, xlabels, ylabels+['Average'], 
                './bg_fg/by_preds/{}'.format(m), ' '.join(['Average'] + m.split('_')), fsr_is_dir=False)
        if bar:
            avgs_by_mtype = dict({fs[i]:avgs[i,-1] for i in range(len(fs))})
            avgs_by_class = dict({fs[i]:avgs[-1,:] for i in range(len(fs))})
            plot_bars(avgs_by_mtype, avgs_by_class, m)

### Qualitative: automatically discovering + viewing extreme cases
def update_running_scores(subset, scores, running_scores, worst=True):
    '''
    utility fn to update running list of best or worst scores
    when worst=True, updates running scores to contain lowest 5 scores, including newer scores
    with worst=False, maintains list of top scores
    '''
    sign = 2*int(worst) - 1
    for i in range(subset.shape[0]):
        pos = -1
        while (pos >= -5) and (sign*scores[i] < sign*running_scores[pos][0]):
            pos -= 1
        # shuffle down
        pos += 1
        new_val = (scores[i], subset[i].cpu())
        while pos < 0:
            old_val = running_scores[pos]
            running_scores[pos] = new_val
            new_val = old_val
            pos += 1
    return running_scores

def extract_extreme_cases(mtype, loader, metric_args):
    ''' 
    Extract examples with best and worst saliency alignment.
    '''
    model, _ = obtain_model(mtype)
    model = model.cuda()

    worst_cases = dict({target:[(1,0)]*5 for target in range(10)})
    best_cases = dict({target:[(0,0)]*5 for target in range(10)})

    stats = load_cached_results(save_root='./bg_fg/by_preds/', key='{}_bg_fg_scores_by_preds'.format(mtype), overwrite=False)
    scores = process_scores(metric_args, stats)
    ctr_by_class = dict({i:0 for i in range(10)})

    for imgs, masks, labels in tqdm(loader):
        imgs = imgs.cuda()
        preds = model(imgs).argmax(1)
        imgs = imgs.detach().cpu()
        for target in range(10):
            if sum(preds == target) == 0:
                continue
            
            subset = imgs[preds == target]
            subset_scores = scores[target][ctr_by_class[target]:ctr_by_class[target]+subset.shape[0]]
            ctr_by_class[target] += subset.shape[0]

            worst_cases[target] = update_running_scores(subset, subset_scores, worst_cases[target], worst=True)
            best_cases[target] = update_running_scores(subset, subset_scores, best_cases[target], worst=False)

    extreme_cases_dict = load_cached_results(save_root='./bg_fg/extreme_cases/', key='{}_bg_fg_extreme_cases'.format(mtype))
    extreme_cases_dict['best_{}'.format(metric_args['key'])] = best_cases
    extreme_cases_dict['worst_{}'.format(metric_args['key'])] = worst_cases
    cache_results(stats, save_root='./bg_fg/extreme_cases/', key='{}_bg_fg_extreme_cases'.format(mtype))

def view_extreme_cases(mtype, metric_key, worst=True):
    '''
    Useful for viewing worst saliency alignment cases -- automatic discovery of spurious bg features
    '''
    target_category = []
    all_imgs = []
    with open('./bg_fg/by_preds/{}_bg_fg_scores_by_preds.pkl'.format(mtype), 'rb') as fp:
        scores = pickle.load(fp)
    for i in range(10):
        l = scores['{}_{}'.format('worst' if worst else 'best', metric_key)][i]
        imgs = torch.stack([x[1] for x in l])
        all_imgs.append(imgs)
        target_category.extend([i]*len(l))
    all_imgs = torch.cat(all_imgs)
    model, target_layer = obtain_model(mtype)
    cam = get_cam_obj(model, target_layer, mtype)
    grid, _, _, _, _, _ = gradcam_vis_and_score(model, all_imgs, torch.zeros(all_imgs.shape), target_layer, target_category, cam=cam)
    grid.save('./visualizations/{}_{}_{}_cases.png'.format(mtype, 'worst' if worst else 'best', metric_key))

if __name__ == '__main__':
    testset = RIVAL10(train=False, return_masks=True)
    loader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False)

    aps = dict({'key':'aps'})
    fracs_inside = dict({'key':'fracs_inside'})
    delta_densities = dict({'key':'delta_densities'})
    mc25 = dict({'key':'mask_coverages_25', 'mc_ind':0})
    ious = dict({'key':'ious'})
    
    # to extract and view worst delta_densities for deit_tiny model
    extract_extreme_cases(mtype='deit_tiny', loader=loader, metric_args=delta_densities)
    view_extreme_cases(mtype='deit_tiny', metric_key='delta_densities', worst=True)

    # to generate saliency alignment plots
    gen_heatmap([ious, aps, delta_densities, fracs_inside, mc25], bar=True)

