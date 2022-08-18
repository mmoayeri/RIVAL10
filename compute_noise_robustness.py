from finetuner import FineTuner
from utils import *
from tqdm import tqdm
import pickle
import numpy as np
from datasets import RIVAL10

epsilon_dict = dict({'linf': [x/255 for x in [30,60,90,120,150,180,210]],
                     'l2' : [25,50,75,100, 125,150,175,200]})

def noise_robustness(mtype, loader, eps=25.0, num_trials=1, linf=False, adv=False, adv_l2=False):
    model, _ = obtain_model(mtype)
    model = model.cuda()

    all_og_probs = dict({target:[] for target in range(10)})
    all_noisy_bg_probs = dict({target:[] for target in range(10)})
    all_noisy_fg_probs = dict({target:[] for target in range(10)})

    og_cc = dict({target:0 for target in range(10)})
    noisy_bg_cc = dict({target:0 for target in range(10)})
    noisy_fg_cc = dict({target:0 for target in range(10)})
    ctrs = dict({target:0 for target in range(10)})

    for imgs, masks, labels in tqdm(loader):
        noise = torch.randn(imgs.shape)
        imgs = imgs.cuda()

        preds = model(imgs).argmax(1)
        imgs = imgs.detach().cpu()

        for target in range(10):
            if sum(labels == target) == 0:
                continue

            running_noisy_bg_probs, running_noisy_fg_probs = [], []
            running_probs = []

            subset = imgs[labels == target].cuda()
            subset_labels = labels[labels == target].cuda()
            subset_masks = masks[labels == target].cuda()

            for trial in range(num_trials):
                noise = torch.randn(subset.shape).cuda()
                fg_noise = noise * subset_masks
                bg_noise = noise * (1-subset_masks)
                if not linf:
                    fg_noise = l2_normalize(fg_noise)
                    bg_noise = l2_normalize(bg_noise)
                    noise = l2_normalize(noise)

                subset_noisy_fg = torch.clamp(subset + fg_noise * eps, 0, 1)
                subset_noisy_bg = torch.clamp(subset + bg_noise * eps, 0, 1)
                subset_noisy = torch.clamp(subset + noise * eps, 0, 1)

                clean_logits = model(subset)
                noisy_bg_logits = model(subset_noisy_bg)
                noisy_fg_logits = model(subset_noisy_fg)

                og_probs = torch.softmax(clean_logits, 1)[:,target].detach().cpu().numpy()
                noisy_bg_probs = torch.softmax(noisy_bg_logits, 1)[:,target].detach().cpu().numpy()
                noisy_fg_probs = torch.softmax(noisy_fg_logits, 1)[:,target].detach().cpu().numpy()

                og_cc[target] += (clean_logits.argmax(1) == subset_labels).sum().item()
                noisy_bg_cc[target] += (noisy_bg_logits.argmax(1) == subset_labels).sum().item()
                noisy_fg_cc[target] += (noisy_fg_logits.argmax(1) == subset_labels).sum().item()
                ctrs[target] += subset_labels.shape[0]

                running_probs.append(og_probs)
                running_noisy_bg_probs.append(noisy_bg_probs)
                running_noisy_fg_probs.append(noisy_fg_probs)

            avg_clean_probs = np.average(running_probs, 0)
            avg_noisy_bg_probs = np.average(running_noisy_bg_probs, 0)          
            avg_noisy_fg_probs = np.average(running_noisy_fg_probs, 0)   

            all_og_probs[target].extend(avg_clean_probs)
            all_noisy_bg_probs[target].extend(avg_noisy_bg_probs)
            all_noisy_fg_probs[target].extend(avg_noisy_fg_probs)

    og_accs = [og_cc[i]/ctrs[i] for i in range(10)]
    noisy_bg_accs = [noisy_bg_cc[i]/ctrs[i] for i in range(10)]
    noisy_fg_accs = [noisy_fg_cc[i]/ctrs[i] for i in range(10)]
    for i in range(10):
        print('Og acc: {:.3f}'.format(og_accs[i]))
        print('Noisy bg acc: {:.3f}, Noisy fg acc: {:.3f}'.format(noisy_bg_accs[i], noisy_fg_accs[i]))


    ext = '_linf' if linf else ''
    stats = load_cached_results(save_root='./bg_fg/noise/', key=mtype+ext, overwrite=False)
    stats[eps] = dict({'og_accs':og_accs, 'og_probs': all_og_probs,
                       'noisy_bg_accs':noisy_bg_accs, 'noisy_fg_accs':noisy_fg_accs,
                       'noisy_bg_probs':all_noisy_bg_probs, 'noisy_fg_probs': all_noisy_fg_probs})
    # cache_results(stats, save_root='./bg_fg/noise/', key=mtype+ext)

def compute_noise_robustness_all_models(mtypes):
    dset = RIVAL10(train=False, return_masks=True)
    loader = torch.utils.data.DataLoader(dset, batch_size=16, shuffle=False)
    for mtype in tqdm(mtypes):
        for eps in epsilon_dict['linf']:
            noise_robustness(mtype, loader, linf=True, eps=eps/255)
        for eps in epsilon_dict['l2']:
            noise_robustness(mtype, loader, linf=False, eps=eps)

if __name__ == '__main__':
    _ALL_MTYPES = ['resnet18', 'resnet50', 'resnet101','resnet152', 'robust_resnet18', 'robust_resnet50', 
               'vit_tiny', 'vit_small', 'vit_base', 'deit_tiny', 'deit_small', 'deit_base', 
               'simclr', 'clip_RN50', 'clip_RN101', 'clip_ViT-B16', 'clip_ViT-B32', 
               'robust_resnet18_eps1', 'robust_resnet50_eps1', 'vit_small32', 'vit_base32']
    compute_noise_robustness_all_models(_ALL_MTYPES)
