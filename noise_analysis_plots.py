from finetuner import FineTuner
from torchvision import transforms
from utils import *
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from scipy.stats import gaussian_kde
from sklearn.metrics import average_precision_score
import numpy as np
from matplotlib import cm
import matplotlib.patches as mpatches

### Preliminaries
groups = dict({
    'DeiTs': (['deit_tiny', 'deit_small', 'deit_base'], 's'),
    'CLIP ViTs': (['clip_ViT-B16', 'clip_ViT-B32'], 's'),
    'ViTs': (['vit_tiny', 'vit_small', 'vit_base'], 's'), 
    'ResNets': (['resnet18', 'resnet50', 'resnet101', 'resnet152'], '^'), 
    'CLIP ResNets': (['clip_RN50', 'clip_RN101'], '^'),
    'SimCLR': (['simclr'], '^'),
    'Robust ResNets': (['robust_resnet18', 'robust_resnet50'], '^')
})
param_counts_dict = dict({
    'resnet18': 11.4,
    'resnet50': 23.9,
    'resnet101': 42.8,
    'resnet152': 58.5,
    'robust_resnet18': 11.4,
    'robust_resnet18_eps1': 11.4,
    'robust_resnet50': 23.9,
    'robust_resnet50_eps1': 23.9,
    'deit_tiny': 5,
    'deit_small': 22,
    'deit_base': 86,
    'deit_distilled_tiny': 5,
    'deit_distilled_small': 22,
    'deit_distilled_base': 86,
    'simclr': 23.9,
    'clip_RN50': 23.9,
    'clip_RN101': 42.8,
    'clip_ViT-B16': 86,
    'clip_ViT-B32': 87,
    'vit_tiny': 5,
    'vit_small': 22,
    'vit_base': 86,
    'vit_small32': 23,
    'vit_base32': 87,
})
cmap = cm.get_cmap('jet')
colors = [cmap(x/len(groups)) for x in range(len(groups))]
model_to_marker_and_marker_size_and_color = dict()
for color,group in zip(colors,groups):
    mtypes, marker = groups[group]
    for mtype in mtypes:
        pcount = param_counts_dict[mtype]
        model_to_marker_and_marker_size_and_color[mtype] = (marker, 1.5*pcount, color)#np.sqrt(pcount))

epsilon_dict = dict({'linf': [x/255 for x in [30,60,90,120,150,180,210]],
                     'l2' : [25,50,75,100, 125,150,175,200]})


_ALL_MTYPES = ['resnet18', 'resnet50', 'resnet101','resnet152', 
            'robust_resnet18', 'robust_resnet50', 'vit_tiny', 'vit_small', 'vit_base', 
            'deit_tiny', 'deit_small', 'deit_base', 'simclr', 'clip_RN50', 'clip_RN101', 
            'clip_ViT-B16', 'clip_ViT-B32']

#### Plot generating code:

def plot_fg_bg_scatter(mtypes, stat='acc', linf=True, n_per_row=6):
    '''
    Generates Figure 2 in paper. 
    Per model, plots scatter of (noisy_fg_acc, noisy_bg_acc), with pt per class.
    Also generates one single plot with a point per model. 
    '''
    ext = '_linf' if linf else ''
    epsilons = epsilon_dict['linf' if linf else 'l2']    
    dcr_idx_to_class_dict = get_dcr_idx_to_class_dict()
    nrows = int(np.ceil(len(mtypes) / n_per_row))
    f, axs = plt.subplots(nrows, n_per_row, figsize=(5*n_per_row, 4*nrows))
    f2, ax2 = plt.subplots(1,1, figsize=(6,4))
    cmap = cm.get_cmap('gist_ncar')
    min_x, min_y = 1, 1
    for j, mtype in enumerate(mtypes):
        ax = axs[int(j// n_per_row), int(j % n_per_row)]
        ax.set_title(mtype)
        d = load_cached_results(save_root='./bg_fg/noise/', key=mtype+ext, overwrite=False)
        fg_stats_by_class, bg_stats_by_class = [np.zeros((10,)) for x in range(2)]
        for i in range(10):
            fg_stat, bg_stat = [], []
            for eps in epsilons:
                if stat == 'acc':
                    fg_stat.append(d[eps]['noisy_fg_{}s'.format(stat)][i])
                    bg_stat.append(d[eps]['noisy_bg_{}s'.format(stat)][i])
                elif stat == 'prob':
                    fg_stat.extend(d[eps]['noisy_fg_{}s'.format(stat)][i])
                    bg_stat.extend(d[eps]['noisy_bg_{}s'.format(stat)][i])
            fg_stats_by_class[i] = np.nanmean(fg_stat)
            bg_stats_by_class[i] = np.nanmean(bg_stat)
            ax.scatter(np.nanmean(fg_stat), np.nanmean(bg_stat), color=cmap(i/10), label=dcr_idx_to_class_dict[i])

        stat_name = 'Accuracy' if stat=='acc' else 'Pr[True Class]'
        ax.set_xlabel('Average {} under Foreground {} Noise'.format(stat_name, '$L_\infty$' if linf else '$L_2$'))
        ax.set_ylabel('Average {} under Background {} Noise'.format(stat_name, '$L_\infty$' if linf else '$L_2$'))
        ax.set_xlim((0,1))
        ax.set_ylim((0,1))
        xs = np.linspace(0,1,100)
        ax.plot(xs, xs, '-')
        # average plot: single plot, 1 pt per model (no more per class)
        marker, size, color = model_to_marker_and_marker_size_and_color[mtype]
        x,y = [np.average(vals) for vals in [fg_stats_by_class, bg_stats_by_class]]
        ax2.scatter(x, y, color=color, label=get_model_name(mtype), marker=marker, s=size, edgecolors='black', linewidths=0.4)
        min_x = min(min_x, x); min_y = min(min_y, y)

    axs[int(nrows / 2), -1].legend(bbox_to_anchor=(1,1))
    f.tight_layout()
    f.savefig('bg_fg/noise/plots/scatter_fg_bg_{}_per_model{}.png'.format(stat, '_linf' if linf else ''))

    ax2.set_xlabel('Average {} under Foreground {} Noise'.format(stat_name, '$L_\infty$' if linf else '$L_2$'))
    ax2.set_ylabel('Average {} under Background {} Noise'.format(stat_name, '$L_\infty$' if linf else '$L_2$'))
    min_xy = min(min_x, min_y)-0.03
    xs = np.linspace(min_xy, 1, 100)
    ax2.plot(xs, xs, '-')
    ax2.set_xlim([min_xy,1]); ax2.set_ylim([min_xy,1])
    ax2.legend(bbox_to_anchor=(1.05,1.05))
    f2.tight_layout()
    f2.savefig('bg_fg/noise/plots/scatter_fg_bg_{}{}.png'.format(stat, '_linf' if linf else ''), dpi=300)

def hist_per_model_bg_v_fg(mtypes, linf=True, n_per_row=5, scatter=False, class_idx_to_display=[*range(10)]):
    '''
    Generates bottom plots in Figure 5 of paper.
    Per model 2d histograms or scatter plots for instance wise model performance.  
    '''
    n_per_row = min(n_per_row, len(mtypes))
    ext = '_linf' if linf else ''
    epsilons = epsilon_dict['linf' if linf else 'l2']
    nrows = int(np.ceil(len(mtypes) / n_per_row))
    dcr_idx_to_class_dict = get_dcr_idx_to_class_dict()
    f, axs = plt.subplots(nrows, n_per_row, figsize=(4*n_per_row, 4*nrows))
    if nrows == 1:
        axs = axs.reshape(1, n_per_row)
    cmap = cm.get_cmap('gist_ncar')
    colors = [cmap(x/len(class_idx_to_display)) for x in range(len(class_idx_to_display))]
    handles = [mpatches.Patch(color=c, label=dcr_idx_to_class_dict[i]) for c,i in zip(colors, class_idx_to_display)]
    hist_per_mtype = dict()
    extra_artists = []
    for j, mtype in enumerate(mtypes):
        ax = axs[int(j// n_per_row), int(j % n_per_row)]
        stats = load_cached_results(save_root='./bg_fg/noise/', key=mtype+ext, overwrite=False)
        all_noisy_bg_probs, all_noisy_fg_probs = [], []
        for eps in epsilons:
            noisy_bg_probs_dict, noisy_fg_probs_dict = [stats[eps]['noisy_{}g_probs'.format(x)] for x in ['b','f']]
            for c,i in zip(colors, class_idx_to_display):
                all_noisy_bg_probs.extend(noisy_bg_probs_dict[i])
                all_noisy_fg_probs.extend(noisy_fg_probs_dict[i])
                if scatter:
                    ax.scatter(noisy_fg_probs_dict[i], noisy_bg_probs_dict[i], color=c, alpha=0.05)
        all_noisy_bg_probs, all_noisy_fg_probs = [np.array(x) for x in [all_noisy_bg_probs, all_noisy_fg_probs]]
        all_noisy_fg_probs[np.isnan(all_noisy_fg_probs)] = 0
        all_noisy_bg_probs[np.isnan(all_noisy_bg_probs)] = 0
        if not scatter:
            nside = 50
            o = ax.hist2d(all_noisy_fg_probs, all_noisy_bg_probs, bins=(np.arange(nside)/nside, np.arange(nside)/nside), density=True)#,vmin=0, vmax=10)
            hist_per_mtype[mtype] = o[0] / np.max(o[0])
        xs = np.linspace(0,1,100)
        ax.plot(xs, xs, '-')
        ax.set_xlabel('Pr[True Class] under Fg Noise')
        ax.xaxis.label.set_size(12)
        ax.set_ylabel('Pr[True Class] under Bg Noise')
        ax.yaxis.label.set_size(12)
        extra_artists.append(ax.set_title(get_model_name(mtype), y=-0.3))
        ax.title.set_size(16)
    f.tight_layout()
    if scatter:
        lg = axs[-1, int(n_per_row / 2)].legend(handles=handles, prop={'size': 14}, loc='upper right')#bbox_to_anchor=(1,-0.2), ncol=len(class_idx_to_display), )
        extra_artists.append(lg)
    plt.savefig('./bg_fg/noise/plots/{}_probs_under_noise{}.png'.format('scatter' if scatter else 'hist', '_linf' if linf else ''), 
            dpi=300, bbox_inches='tight', bbox_extra_artists=extra_artists)
    if not scatter:
        combine_hists(hist_per_mtype, groups, linf)            

def plot_noise_robs(linf=True, plot2_stat='normed_diff'):
    '''
    Generates Figure 4 of paper.
    Plots acc under fg noise, acc under bg noise, and some version of the gap
    in three subplots. normed_diff corresponds to using RFS. 
    '''
    plt.close();plt.clf()
    f, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(12,4),dpi=300)
    epsilons = epsilon_dict['linf' if linf else 'l2']
    ext = '_linf' if linf else ''
    handles = []
    for c, group in zip(colors, groups):
        all_fg_accs, all_bg_accs = [], []
        mtypes, symbol = groups[group]
        dash_type = '-' if symbol == 's' else '--'
        for mtype in mtypes:
            stats = load_cached_results(save_root='./bg_fg/noise/', key=mtype+ext, overwrite=False)
            fg_accs = [np.average(stats[eps]['noisy_fg_accs']) for eps in epsilons]
            bg_accs = [np.average(stats[eps]['noisy_bg_accs']) for eps in epsilons]
            all_fg_accs.append(fg_accs); all_bg_accs.append(bg_accs)
        fg_accs = np.average(all_fg_accs, axis=0)
        bg_accs = np.average(all_bg_accs, axis=0)
        handles.append(mpatches.Patch(color=c, label=group))
        ax1.plot(epsilons, fg_accs, dash_type, color=c, label=group)
        ax2.plot(epsilons, bg_accs, dash_type, color=c, label=group)
        if plot2_stat == 'relative_diff':
            diff = (bg_accs - fg_accs) / bg_accs
            ylabel = 'Relative Difference in Accuracy under FG and BG Noise'
        elif plot2_stat == 'raw_diff':
            diff = bg_accs - fg_accs
            ylabel = 'Gap in Accuracy under FG and BG Noise'
        elif plot2_stat == 'normed_diff':
            diff = bg_accs - fg_accs
            diff = diff / np.min((0.5*(fg_accs+bg_accs), 1-0.5*(fg_accs+bg_accs)),0) / 2
            ylabel = '$RFS$'#'Normalized Gap in Accuracy under FG and BG Noise'
        ax3.plot(epsilons, diff, dash_type, color=c, label=group)
    ax2.legend()
    xlabel = 'Standard deviation $\sigma$ of Gaussian Noise' if linf else 'Norm of $L_2$ Normalized Gaussian Noise'
    ax1.set_xlabel(xlabel)
    ax2.set_xlabel(xlabel)
    ax1.xaxis.label.set_size(10); ax1.yaxis.label.set_size(10)
    ax1.set_ylim([0.3,1])
    ax2.set_ylim([0.3,1])
    ax1.set_ylabel('Accuracy under {} Noise in Foreground'.format('$L_\infty$' if linf else '$L_2$'))
    ax2.set_ylabel('Accuracy under {} Noise in Background'.format('$L_\infty$' if linf else '$L_2$'))
    ax2.xaxis.label.set_size(10); ax2.yaxis.label.set_size(10)
    ax1.set_title('Foreground Noise')
    ax2.set_title('Background Noise')
    ax3.set_title('Relative Foreground Sensitivity')
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel(ylabel)
    ax3.xaxis.label.set_size(10); ax3.yaxis.label.set_size(10)
    f.tight_layout()
    f.savefig('./bg_fg/noise/plots/accs_under_noise{}_{}.png'.format('_linf' if linf else '', plot2_stat), dpi=300)

def kde_hist(x, ax, label, color):
    xx = np.linspace(-1,1, 500)
    kde = gaussian_kde(x)
    ax.plot(xx, kde(xx), label=label, color=color)

def hist_normed_sensitivities(mtypes, linf=True, norm_type='normalize', class_idx_to_display=[1,4,3,9]):
    ''' 
    Generates top row of figure 5 in paper.
    Per model, we generate a subplot that overlays smoothed histograms of iRFS per class.
    '''
    plt.clf()
    ext = '_linf' if linf else '' 
    epsilons = epsilon_dict['linf' if linf else 'l2']
    cmap = cm.get_cmap('gist_ncar')
    colors = [cmap(x/len(class_idx_to_display)) for x in range(len(class_idx_to_display))]
    f, axs = plt.subplots(1, len(mtypes), figsize=(4*len(mtypes), 4))#, sharey=True)
    dcr_idx_to_class_dict = get_dcr_idx_to_class_dict()
    axs[0].set_ylabel('Frequency')
    for ax, mtype in zip(axs, mtypes):
        stats = load_cached_results(save_root='./bg_fg/noise/', key=mtype+ext, overwrite=False)
        all_diffs = dict({i:[] for i in range(10)})
        for eps in epsilons:
            for i in range(10):
                fg_probs = np.array(stats[eps]['noisy_fg_probs'][i])
                bg_probs = np.array(stats[eps]['noisy_bg_probs'][i])
                diffs = bg_probs - fg_probs
                if norm_type == 'normalize':
                    diffs = diffs / np.min((0.5*(fg_probs+bg_probs), 1-0.5*(fg_probs+bg_probs)),0) / 2
                elif norm_type == 'relative':
                    diffs = diffs / np.maximum(fg_probs, bg_probs)
                diffs[np.isnan(diffs)] = 0
                all_diffs[i].extend(diffs)
        for idx, c in zip(class_idx_to_display, colors):
            kde_hist(all_diffs[idx], ax, label=dcr_idx_to_class_dict[idx], color=c)
        
        ax.set_xlabel('iRFS')
        ax.set_ylabel('Frequency')
        ax.xaxis.label.set_size(12); ax.yaxis.label.set_size(12)
        if ax == axs[len(mtypes) // 2]:
            ax.legend(prop={'size':14})
        ax.set_ylim([0,5])
    
    f.tight_layout()
    f.savefig('./bg_fg/noise/plots/hist_prob_diffs{}_{}.png'.format('_linf' if linf else '', norm_type), 
            dpi=500, bbox_inches='tight')

def update_running_scores(subset, scores, fg_probs, subset_masks, running_scores, worst=True):
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
        new_val = (scores[i], fg_probs[i], subset_masks[i].cpu(), subset[i].cpu())
        while pos < 0:
            old_val = running_scores[pos]
            running_scores[pos] = new_val
            new_val = old_val
            pos += 1
    return running_scores

def view_extreme_cases(mtypes, eps=150, linf=True, worst=True):
    '''
    Generates Figure 1 in paper.
    '''
    # mtypes = ['robust_resnet50', 'simclr','deit_tiny','deit_small', 'deit_base']
    ext = '_linf' if linf else '' 
    dcr_idx_to_class_dict = get_dcr_idx_to_class_dict()

    for mtype in tqdm(mtypes):
        target_category = []
        all_imgs = []
        root_path = './visualizations/noise/'+mtype+ext+'/eps_{}/'.format(eps)
        os.makedirs(root_path, exist_ok=True)
        ft = FineTuner(mtype)
        ft.restore_model()
        acc = 100*ft.best_acc
        with open('./noise_extreme_cases/{}.pkl'.format(mtype+ext), 'rb') as fp:
            scores = pickle.load(fp)
            print(scores.keys())
        for i in range(10):
            l = scores[eps]['{}_{}'.format('worst' if worst else 'best', 'noise_gaps')][i]
            gap, fg_prob, mask, img = l[0]
            img, mask = [torch.tensor(x) for x in [img, mask]]
            noise = torch.randn(img.shape) * eps / 255
            noisy_bg = torch.clamp((img + (1-mask) * noise), 0, 1).numpy()
            noisy_fg = torch.clamp((img + mask * noise), 0, 1).numpy()

            f, (ax1, ax2) = plt.subplots(1,2, figsize=(6,3))
            ax1.imshow(noisy_bg.swapaxes(0,1).swapaxes(1,2))
            class_name = dcr_idx_to_class_dict[i]
            ax1.text(30, 240, 'Pr[{}] = {:.1f}%'.format(class_name, 100*(fg_prob+gap)), size=12)
            ax1.set_xticks([]); ax1.set_yticks([])
            ax2.imshow(noisy_fg.swapaxes(0,1).swapaxes(1,2))
            ax2.text(30, 240, 'Pr[{}] = {:.1f}%'.format(class_name, 100*(fg_prob)), size=12)
            ax2.set_xticks([]); ax2.set_yticks([])
            st = f.suptitle('{}, {:.1f}% accuracy on clean samples'.format(get_model_name(mtype), acc), y=0.92)
            f.savefig('{}/{}.png'.format(root_path, class_name), dpi=300, bbox_inches='tight', extra_artists=[st])

def extract_extreme_cases(mtype, loader, linf=True, eps=150):
    '''
    Method for extracting extreme cases where bg noise degrades model performance much more than fg noise.
    '''
    worst_cases = dict({target:[(1,0,0)]*5 for target in range(10)})
    ext = '_linf' if linf else '' 
    d = load_cached_results(save_root='./bg_fg/noise/', key=mtype+ext, overwrite=False)
    noisy_bg_probs = d[eps/255]['noisy_bg_probs']
    noisy_fg_probs = d[eps/255]['noisy_fg_probs']
    ctr_by_class = dict({i:0 for i in range(10)})

    for imgs, masks, labels in tqdm(loader):
        for target in range(10):
            if sum(labels == target) == 0:
                continue
            
            subset = imgs[labels == target]
            subset_masks = masks[labels == target]
            subset_scores = np.array(noisy_bg_probs[target][ctr_by_class[target]:ctr_by_class[target]+subset.shape[0]]) \
                            - np.array(noisy_fg_probs[target][ctr_by_class[target]:ctr_by_class[target]+subset.shape[0]])
            fg_probs = noisy_fg_probs[target][ctr_by_class[target]:ctr_by_class[target]+subset.shape[0]]
            ctr_by_class[target] += subset.shape[0]

            worst_cases[target] = update_running_scores(subset, subset_scores, fg_probs, subset_masks, worst_cases[target], worst=True)

    stats = load_cached_results(save_root='./noise_extreme_cases/', key=mtype+ext)
    results_dict = dict({'worst_noise_gaps': worst_cases})
    stats[eps] = results_dict
    cache_results(stats, save_root='./noise_extreme_cases/', key=mtype+ext)

def vit_patchsize_effect(f, ax, linf=True):
    '''
    Generates left plot of figure 6 in paper.
    '''
    patchsize_groups = dict({'ViT (Small)': ['vit_small', 'vit_small32'],
                             'ViT (Base)': ['vit_base', 'vit_base32'],
                             'CLIP ViT (Base)': ['clip_ViT-B16', 'clip_ViT-B32']})

    ext = '_linf' if linf else '' 
    epsilons = epsilon_dict['linf' if linf else 'l2']
    cmap = cm.get_cmap('jet')
    c1, c2 = cmap(0.2), cmap(0.8)
    xlabels = []
    for x, g in enumerate(patchsize_groups):
        mtype16, mtype32 = patchsize_groups[g]
        d16, d32 = [load_cached_results(save_root='./bg_fg/noise/', key=mtype+ext, overwrite=False)
                        for mtype in [mtype16, mtype32]]
        all_fg_accs_16, all_bg_accs_16 = [],[]
        all_fg_accs_32, all_bg_accs_32 = [],[]
        for eps in epsilons:
            fg_accs_16 = [np.nanmean(d16[eps]['noisy_fg_accs']) for eps in epsilons]
            bg_accs_16 = [np.nanmean(d16[eps]['noisy_bg_accs']) for eps in epsilons]
            all_fg_accs_16.append(fg_accs_16); all_bg_accs_16.append(bg_accs_16)

            fg_accs_32 = [np.nanmean(d32[eps]['noisy_fg_accs']) for eps in epsilons]
            bg_accs_32 = [np.nanmean(d32[eps]['noisy_bg_accs']) for eps in epsilons]
            all_fg_accs_32.append(fg_accs_32); all_bg_accs_32.append(bg_accs_32)
        
        fg_acc16, bg_acc16 = [np.nanmean(x) for x in [all_fg_accs_16, all_bg_accs_16]]
        fg_acc32, bg_acc32 = [np.nanmean(x) for x in [all_fg_accs_32, all_bg_accs_32]]
        print(fg_acc16, bg_acc16, fg_acc32, bg_acc32)
        norm_diff16 = (bg_acc16 - fg_acc16) / min(0.5*(bg_acc16+fg_acc16), 1-0.5*(bg_acc16+fg_acc16)) / 2
        norm_diff32 = (bg_acc32 - fg_acc32) / min(0.5*(bg_acc32+fg_acc32), 1-0.5*(bg_acc32+fg_acc32)) / 2

        ax.bar(x-0.16, norm_diff16, width=0.32, color=c1)
        ax.bar(x+0.16, norm_diff32, width=0.32, color=c2)
        xlabels.append(g)

    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_ylabel('Average $RFS$ over All Noise Levels')
    ax.set_title('Effect of Patch Size in ViTs')
    handles = [mpatches.Patch(color=c1, label='Patch=16x16'), mpatches.Patch(color=c2, label='Patch=32x32')]
    ax.legend(handles=handles, loc='lower right')
    f.tight_layout()
    f.savefig('./bg_fg/noise/plots/patchsize{}.png'.format(ext), dpi=300, tight_layout=True)
    

def adv_training_effect(f, ax, linf=True):
    '''
    Generates right plot of Figure 6 in paper. 
    '''
    patchsize_groups = dict({'ResNet18': ['resnet18', 'robust_resnet18_eps1', 'robust_resnet18'],
                             'ResNet50': ['resnet50', 'robust_resnet50_eps1', 'robust_resnet50']})

    ext = '_linf' if linf else '' 
    epsilons = epsilon_dict['linf' if linf else 'l2']
    cmap = cm.get_cmap('jet')
    c1, c2, c3 = cmap(0.1), cmap(0.5), cmap(0.9)
    xlabels = []
    for x, g in enumerate(patchsize_groups):
        mtype0, mtype_eps1, mtype_eps3 = patchsize_groups[g]
        eps0, eps1, eps3 = [load_cached_results(save_root='./bg_fg/noise/', key=mtype+ext, overwrite=False)
                            for mtype in [mtype0, mtype_eps1, mtype_eps3]]
        all_fg_accs_0, all_bg_accs_0 = [],[]
        all_fg_accs_1, all_bg_accs_1 = [],[]
        all_fg_accs_3, all_bg_accs_3 = [],[]
        for eps in epsilons:
            fg_accs_0 = [np.nanmean(eps0[eps]['noisy_fg_accs']) for eps in epsilons]
            bg_accs_0 = [np.nanmean(eps0[eps]['noisy_bg_accs']) for eps in epsilons]
            all_fg_accs_0.append(fg_accs_0); all_bg_accs_0.append(bg_accs_0)

            fg_accs_1 = [np.nanmean(eps1[eps]['noisy_fg_accs']) for eps in epsilons]
            bg_accs_1 = [np.nanmean(eps1[eps]['noisy_bg_accs']) for eps in epsilons]
            all_fg_accs_1.append(fg_accs_1); all_bg_accs_1.append(bg_accs_1)

            fg_accs_3 = [np.nanmean(eps3[eps]['noisy_fg_accs']) for eps in epsilons]
            bg_accs_3 = [np.nanmean(eps3[eps]['noisy_bg_accs']) for eps in epsilons]
            all_fg_accs_3.append(fg_accs_3); all_bg_accs_3.append(bg_accs_3)
        
        fg_acc0, bg_acc0 = [np.nanmean(x) for x in [all_fg_accs_0, all_bg_accs_0]]
        fg_acc1, bg_acc1 = [np.nanmean(x) for x in [all_fg_accs_1, all_bg_accs_1]]
        fg_acc3, bg_acc3 = [np.nanmean(x) for x in [all_fg_accs_3, all_bg_accs_3]]

        norm_diff0 = (bg_acc0 - fg_acc0) / min(0.5*(bg_acc0+fg_acc0), 1-0.5*(bg_acc0+fg_acc0)) / 2
        norm_diff1 = (bg_acc1 - fg_acc1) / min(0.5*(bg_acc1+fg_acc1), 1-0.5*(bg_acc1+fg_acc1)) / 2
        norm_diff3 = (bg_acc3 - fg_acc3) / min(0.5*(bg_acc3+fg_acc3), 1-0.5*(bg_acc3+fg_acc3)) / 2

        ax.bar(x-0.2, norm_diff0, width=0.2, color=c1)
        ax.bar(x, norm_diff1, width=0.2, color=c2)
        ax.bar(x+0.2, norm_diff3, width=0.2, color=c3)
        xlabels.append(g)

    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_ylabel('Average $RFS$ over All Noise Levels')
    ax.set_title('Effect of $\epsilon$ in Adv. Training')
    handles = [mpatches.Patch(color=c1, label='Standard'), mpatches.Patch(color=c2, label='Robust ($\epsilon=1$)'), 
               mpatches.Patch(color=c3, label='Robust ($\epsilon=3$)')]
    ax.legend(handles=handles, loc='upper left')
    f.tight_layout()
    f.savefig('./bg_fg/noise/plots/robustness{}.png'.format(ext), dpi=300, tight_layout=True)

if __name__ == '__main__':
    testset = RIVAL10(train=False, return_masks=True)
    loader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False)

    # Generate Figure 1
    mtypes = ['robust_resnet50', 'simclr','deit_tiny','deit_small', 'deit_base']
    for mtype in mtypes:
        extract_extreme_cases(mtype, loader, eps=60)
    view_extreme_cases(mtypes, eps=60)

    # Generate Figure 2
    plot_fg_bg_scatter(_ALL_MTYPES, stat='acc', linf=True)

    # Generate Figure 4
    plot_noise_robs(linf=True, plot2_stat='normed_diff')

    # Generate Figure 5
    mtypes2 = ['resnet50', 'robust_resnet50', 'simclr', 'vit_small', 'deit_small']
    hist_normed_sensitivities(mtypes2)
    hist_per_model_bg_v_fg(mtypes2, linf=True, scatter=True, class_idx_to_display=[1,4,3,9]))    

    # Generate Figure 6
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    vit_patchsize_effect(f, ax1)
    adv_training_effect(f, ax2)
