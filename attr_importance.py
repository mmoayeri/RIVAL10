from finetuner import FineTuner
from utils import *
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from scipy.stats import gaussian_kde
import numpy as np
from matplotlib import cm
from datasets.attr_dataset import AttrDataset

def attr_importance(mtype, attr, loader, fg_only=False):
    model,_ = obtain_model(mtype, fg_only=fg_only)
    all_og_probs = dict({i:[] for i in range(10)})
    all_no_attr_probs = dict({i:[] for i in range(10)})
    og_cc, no_attr_cc, ctr = [dict({i:0 for i in range(10)}) for j in range(3)]
    for pos, neg, masks, eo_masks, labels in tqdm(loader):
        if fg_only:
            pos = pos * eo_masks
        pos = pos.cuda()
        og_logits = model(pos)
        og_preds = og_logits.argmax(1)

        for i in range(10):
            subset = pos[labels == i]
            if subset.shape[0] == 0:
                continue
            subset_logits = og_logits[labels == i]
            subset_masks = masks[labels == i].cuda()
            subset_labels = labels[labels == i].cuda()

            gray = 0.5*torch.ones(subset.shape).cuda()
            no_attr = subset * (1-subset_masks) + gray * subset_masks
            no_attr_logits = model(no_attr)

            og_probs, no_attr_probs = [torch.softmax(logits, 1)[:,i] for logits in [subset_logits, no_attr_logits]]   

            og_cc[i] += (subset_logits.argmax(1) == subset_labels).sum().item()
            no_attr_cc[i] += (no_attr_logits.argmax(1) == subset_labels).sum().item()
            ctr[i] += subset.shape[0]

            all_og_probs[i].extend(og_probs.detach().cpu().numpy())
            all_no_attr_probs[i].extend(no_attr_probs.detach().cpu().numpy())

    dcr_idx_to_class_dict = get_dcr_idx_to_class_dict()
    for i in range(10):
        classname = dcr_idx_to_class_dict[i]
        if ctr[i] == 0:
            print('No {}s with a {}'.format(classname, attr))
        else:
            print('For {}s, Og accuracy: {:.3f} ({}/{}). Accuracy after removing attribute: {:.3f} ({}/{})'
                .format(classname, og_cc[i]/ctr[i], og_cc[i], ctr[i], no_attr_cc[i]/ctr[i], no_attr_cc[i], ctr[i]))
    
    key = '{}_{}_ablate_gray{}'.format(mtype, attr, '_fg_only' if fg_only else '')
    d = load_cached_results(save_root='./attr_importance/stats/', key=key, overwrite=False)
    d['og_probs'] = all_og_probs
    d['no_attr_probs'] = all_no_attr_probs
    d['og_cc_stats'] = [(og_cc[i], ctr[i]) for i in range(10)]
    d['no_attr_cc_stats'] = [(no_attr_cc[i], ctr[i]) for i in range(10)]
    cache_results(d, save_root='./attr_importance_dir/stats/', key=key)

def plot_attr_importance_hmap(attrs, mtypes, fg_only=False, probs=False):
    vals = np.zeros((len(attrs)+1, len(mtypes)+1))
    for i, attr in enumerate(attrs):
        for j, mtype in enumerate(mtypes):
            key = '{}_{}_ablate_gray{}'.format(mtype, attr, '_fg_only' if fg_only else '')
            d = load_cached_results(save_root='./attr_importance_dir/stats/', key=key, overwrite=False)
            if probs:
                og_probs = np.concatenate([d['og_probs'][i] for i in range(10)])
                no_attr_probs = np.concatenate([d['no_attr_probs'][i] for i in range(10)])
                avg_diff = np.average(og_probs - no_attr_probs)
                vals[i,j] = 100 * avg_diff
            else:
                og_stats = d['og_cc_stats']
                no_attr_stats = d['no_attr_cc_stats']
                tot = sum([x[1] for x in og_stats])
                og_acc, no_attr_acc = [sum([x[0] for x in ccs]) / tot for ccs in [og_stats, no_attr_stats]]
                vals[i,j] = 100* (og_acc - no_attr_acc)
        vals[i,-1] = np.average(vals[i,:-1])
    vals[-1,:] = np.average(vals[:-1,:], 0)
    # print(vals.shape, len(attrs+['Average']))
    mnames = [get_model_name(m) for m in mtypes]
    heatmap(vals, mnames+['Average'], attrs+['Average'], 
            './attr_importance_dir/{}_drops_attrs_by_mtypes{}'.format('prob' if probs else 'acc', '_fg_only' if fg_only else ''), 
            '{} Drop after Ablating Attribute{}'.format('Pr[True Class]' if probs else 'Accuracy', '(FG only)' if fg_only else ''), fsr_is_dir=False)

def plot_model_accs_bg_ablation(mtypes):
    normal_model_fg_only_accs = []
    fg_only_model_fg_only_accs = []
    d = load_cached_results(save_root='./attr_importance_dir/stats', key='acc_bg_ablated', overwrite=False)
    for mtype in mtypes:
        if mtype not in d:
            ft = FineTuner(mtype, fix_ftrs=True, fg_only=False)
            ft.restore_model()
            ft.fg_only = True; ft.init_loaders()
            _, normal_model_fg_only_acc = ft.process_epoch('test')
            d[mtype] = normal_model_fg_only_acc
        normal_model_fg_only_accs.append(d[mtype])
        
        if mtype+'_fg_only' not in d:
            ft_fg_only = FineTuner(mtype, fix_ftrs=True, fg_only=True)
            ft_fg_only.restore_model()
            # _, fg_only_model_fg_only_acc = ft_fg_only.process_epoch('test')
            d[mtype+'_fg_only'] = ft_fg_only.best_acc
        fg_only_model_fg_only_accs.append(d[mtype+'_fg_only'])
    cache_results(d, save_root='./attr_importance_dir/stats', key='acc_bg_ablated')

    f, (ax1, ax2) = plt.subplots(1,2, figsize=(9,5), sharey=True)
    xs = 1 + np.arange(len(mtypes))
    ax1.bar(xs, normal_model_fg_only_accs)
    ax1.set_xticks(xs); ax1.set_xticklabels(mtypes, rotation='vertical')
    ax2.set_xticks(xs); ax2.set_xticklabels(mtypes, rotation='vertical')
    ax1.set_ylabel('Accuracy on Images with BG Ablated')
    ax2.bar(xs, fg_only_model_fg_only_accs)
    ax1.set_title('Models finetuned on Normal Images')
    ax2.set_title('Models finetuned on BG-Ablated Images')
    f.tight_layout()
    f.savefig('./attr_importance_dir/model_acc_on_fg_only.png', tight_layout=True)

def plot_bg_ablated_accs_grouped(groups, colors):
    '''
    Generates figure 8 in paper.
    Computes accuracy on bg_ablated images. Plots results as bar plot.
    '''
    d = load_cached_results(save_root='./attr_importance_dir/stats', key='acc_bg_ablated', overwrite=False)
    accs_by_group, fg_only_accs_by_group = [], []
    for g in groups:
        accs = []
        fg_only_accs = []
        for mtype in groups[g][0]:
            accs.append(d[mtype].item())
            fg_only_accs.append(d[mtype+'_fg_only'].item())
        accs_by_group.append(np.average(accs))
        fg_only_accs_by_group.append(np.average(fg_only_accs))
    f, axs = plt.subplots(1,2, figsize=(6,3), sharey=True)
    xs = np.arange(len(groups))
    group_names = list(groups.keys())
    for gname, acc, fg_only_acc, x, c in zip(group_names, accs_by_group, fg_only_accs_by_group, xs, colors):
        axs[0].bar(x, acc, color=c)
        axs[1].bar(x, fg_only_acc, color=c)
    axs[0].set_ylabel('Accuracy on BG Ablated Images')
    axs[0].set_title('Finetuned on \nNon-ablated Images')
    axs[0].set_xticks(xs)
    axs[0].set_xticklabels(group_names, rotation='vertical')
    # axs[1].bar(xs, fg_only_accs_by_group, colors=colors)
    axs[1].set_title('Finetuned on \nBG Ablated Images')
    axs[1].set_xticks(xs)
    axs[1].set_xticklabels(group_names, rotation='vertical')
    # axs[1].legend()
    f.tight_layout()
    f.savefig('./bg_fg/bg_ablation.png', dpi=300)

if __name__ == '__main__':
    groups = dict({
        'DeiTs': (['deit_tiny', 'deit_small', 'deit_base'], 's'),
        'CLIP\nViTs': (['clip_ViT-B16', 'clip_ViT-B32'], 's'),
        'ViTs': (['vit_tiny', 'vit_small', 'vit_base'], 's'), # 
        'ResNets': (['resnet18', 'resnet50', 'resnet101', 'resnet152'], '^'), 
        'CLIP\nResNets': (['clip_RN50', 'clip_RN101'], '^'),
        'SimCLR': (['simclr'], '^'),
        'Robust\nResNets': (['robust_resnet18', 'robust_resnet50'], '^')})
    cmap = cm.get_cmap('jet')
    colors = [cmap(x/len(groups)) for x in range(len(groups))]

    # plot_bg_ablated_accs_grouped(groups, colors)

    attrs=['long-snout', 'wings','wheels', 'text', 'horns', 'floppy-ears','ears', 'colored-eyes', 'tail', 'mane', 'beak']
    _ALL_MTYPES = ['resnet18', 'resnet50', 'resnet101','resnet152', 'robust_resnet18', 'robust_resnet50',
                'deit_tiny', 'deit_small', 'deit_base', 'vit_tiny', 'vit_small', 'vit_base',
                'simclr', 'clip_RN50', 'clip_RN101', 'clip_ViT-B16', 'clip_ViT-B32']

    # Generates Figure 8 in paper (Accuracy on ablated BGs for both normal and fg_only models)
    plot_model_accs_bg_ablation(_ALL_MTYPES)

    # Generates attribute ablation plot in supplementary
    for attr in tqdm(attrs):
        dset = AttrDataset(attr, train=False, return_label_and_eo_mask=True)
        loader = torch.utils.data.DataLoader(dset, batch_size=8, shuffle=False)
        for mtype in tqdm(mtypes):
            print('\n Model type: {}, Attribute: {}, Attribute Importances:'.format(mtype, attr))
            attr_importance(mtype, attr, loader, fg_only=False)

    plot_attr_importance_hmap(attrs, _ALL_MTYPES, fg_only=False)



