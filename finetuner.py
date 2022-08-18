import torch
import torch.nn as nn
import os
from datasets import RIVAL10
from torch.utils.data import DataLoader
from tqdm import tqdm
from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model
import argparse
import timm
from torchvision import models

device = 'cuda' if torch.cuda.is_available() else 'cpu'
_FT_ROOT = '/cmlscratch/mmoayeri/dcr_models/finetuned/'
_IMAGENET_ROOT = '/scratch1/shared/datasets/ILSVRC2012/'
_DEIT_ROOT = '/nfshomes/mmoayeri/.cache/torch/hub/facebookresearch_deit_main'

class FineTuner(object):
    def __init__(self, mtype='resnet50', fix_ftrs=True, fg_only=False, epochs=20):
        self.mtype = mtype
        self.init_model(mtype, fix_ftrs)
        self.optimizer = torch.optim.Adam(self.parameters, lr=0.00001, 
                                    betas=(0.9,0.999), weight_decay=0.0001)
        self.fg_only = fg_only
        save_path = mtype
        save_path += '_ftrs_fixed' if fix_ftrs else ''
        save_path += '_fg_only' if fg_only else ''
        self.save_path = os.path.join(_FT_ROOT, '{}.pth'.format(save_path))
        self.init_loaders()
        self.criterion = nn.CrossEntropyLoss()
        self.best_acc = 0
        self.num_epochs = epochs

    def init_model(self, mtype, fix_ftrs):
        if 'deit_distilled' in self.mtype:
            class DistilledDeiTWrapper(nn.Module):
                def __init__(self, mtype):
                    super(DistilledDeiTWrapper, self).__init__()
                    if 'tiny' in mtype:
                        model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)
                    elif 'small' in mtype:
                        model = torch.hub.load('facebookresearch/deit:main', 'deit_small_distilled_patch16_224', pretrained=True)
                    elif 'base' in mtype:
                        model = torch.hub.load('facebookresearch/deit:main', 'deit_base_distilled_patch16_224', pretrained=True)
                    nftrs = model.head.in_features
                    model.head = nn.Linear(in_features=nftrs, out_features=10, bias=True)
                    model.head_dist = nn.Linear(in_features=nftrs, out_features=10, bias=True)
                    self.feat_net = torch.nn.Sequential(model.patch_embed, model.pos_drop, model.blocks, model.norm)
                    self.classifier = torch.nn.Sequential(model.head, model.head_dist) # ya this is weird but don't worry
                    self.model = model
                
                def forward(self, x):
                    out = self.model(x)
                    # i'm embarrased
                    if type(out) is tuple:
                        out = (out[0]+out[1]) / 2
                    return out

            model = DistilledDeiTWrapper(self.mtype)
            feat_net = model.feat_net
            classifier = model.classifier 
            self.gradcam_layer = model.model.blocks[-1].norm1

        elif 'deit' in self.mtype:
            if 'small' in self.mtype:
                model = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True, source='local')
            elif 'base' in self.mtype:
                model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
                # model = torch.hub.load(_DEIT_ROOT, 'deit_base_patch16_224', pretrained=True, source='local')
            elif 'tiny' in self.mtype:
                model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
            # model = torch.hub.load(_DEIT_ROOT, 'deit_base_patch16_224', pretrained=True, source='local')

            nftrs = model.head.in_features
            model.head = nn.Linear(in_features=nftrs, out_features=10, bias=True)
            self.gradcam_layer = model.blocks[-1].norm1
            # match our notation later
            model.feat_net = torch.nn.Sequential(model.patch_embed, model.pos_drop, model.blocks, model.norm)
            model.classifier = model.head

        elif 'vit' in self.mtype:
            patchsize = '32' if '32' in self.mtype else 16
            if 'small' in self.mtype:
                model = timm.create_model('vit_small_patch{}_224'.format(patchsize), pretrained=True)
            elif 'base' in self.mtype:
                model = timm.create_model('vit_base_patch{}_224'.format(patchsize), pretrained=True)
            elif 'tiny' in self.mtype:
                model = timm.create_model('vit_tiny_patch{}_224'.format(patchsize), pretrained=True)
            
            self.gradcam_layer = model.blocks[-1].norm1
            nftrs = model.head.in_features
            model.head = nn.Linear(in_features=nftrs, out_features=10, bias=True)
            model.feat_net = torch.nn.Sequential(model.patch_embed, model.pos_drop, model.blocks, model.norm)
            model.classifier = model.head

        elif self.mtype == 'simclr':

            class SimCLRWrapper(nn.Module):
                def __init__(self):
                    super(SimCLRWrapper, self).__init__()
                    from pl_bolts.models.self_supervised import SimCLR
                    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
                    simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
                    # simclr.freeze()
                    self.model = simclr.encoder
                    nftrs = self.model.fc.in_features
                    self.model.fc = nn.Linear(in_features=nftrs, out_features=10, bias=True)
                    # print(self.model)
                
                def forward(self, x):
                    ftrs = self.model(x)[0]
                    logits = self.model.fc(ftrs)
                    return logits

            # match our notation later
            model = SimCLRWrapper()
            model.feat_net = nn.Sequential(*list(model.model.children())[:-1])
            model.classifier = model.model.fc
            self.gradcam_layer = model.model.layer4[-1]

        elif 'clip' in mtype:
            assert ('ViT' in mtype or 'RN' in mtype), 'CLIP is only supported on ViT-B/16, ViT-B/32, RN50'
            import clip
            clip_mtype = mtype.split('clip_')[-1]
            if 'ViT' in clip_mtype:
                clip_mtype = 'ViT-B/'+clip_mtype[-2:]

            class CLIPWrapper(nn.Module):
                def __init__(self, mtype):
                    super(CLIPWrapper, self).__init__()
                    self.feat_net, self.preprocess = clip.load(mtype, device='cuda')
                    in_ftrs = self.feat_net.encode_image(torch.rand(5,3,224,224).cuda()).shape[1]
                    # in_ftrs =  512 if 'ViT' in mtype else 1024
                    self.classifier = nn.Linear(in_features=in_ftrs, out_features=10, bias=True)

                def forward(self, x):
                    # img_ftrs = self.feat_net.encode_image(self.preprocess(x))
                    img_ftrs = self.feat_net.encode_image(x).float()
                    logits = self.classifier(img_ftrs)
                    return logits

            model = CLIPWrapper(clip_mtype)
            if 'ViT' in clip_mtype:
                self.gradcam_layer = model.feat_net.visual.transformer.resblocks[-1].ln_1
            elif 'RN' in clip_mtype:
                self.gradcam_layer = model.feat_net.visual.layer4[-1]

        elif 'robust' in self.mtype:
            if 'eps' in self.mtype:
                eps = self.mtype.split('eps')[-1]
            else:
                eps = 3
            # arch = 'resnet50' if '50' in self.mtype else 'resnet18'
            ds_ = ImageNet(_IMAGENET_ROOT)
            if 'resnet50' in self.mtype:
                # checkpoint_fp = "/cmlscratch/mmoayeri/dcr_models/pretrained-robust/imagenet_l2_{}_0.pt".format(eps)
                checkpoint_fp = "/cmlscratch/mmoayeri/dcr_models/pretrained-robust/resnet50_l2_eps{}.ckpt".format(eps)
                arch = 'resnet50'
                inftrs = 2048
            elif 'resnet18' in self.mtype:
                checkpoint_fp = "/cmlscratch/mmoayeri/dcr_models/pretrained-robust/resnet18_l2_eps{}.ckpt".format(eps)
                arch = 'resnet18'
                inftrs = 512
            else:
                raise ValueError("only robust resnet50 and resnet18 supported")
            model, _ = make_and_restore_model(arch=arch, dataset=ds_,
                        resume_path=checkpoint_fp)
            self.gradcam_layer = model.model.layer4[-1]
            feat_net = nn.Sequential(*list(model.model.children())[:-1])
            classifier = nn.Linear(in_features=inftrs, out_features=10, bias=True)
            model = nn.Sequential()
            model.add_module('feat_net', feat_net)
            model.add_module('flatten', nn.Flatten())
            model.add_module('classifier', classifier)
            
        else:
            # nonrobust resnet18 and resnet50
            if mtype=='resnet18':
                resnet = models.resnet18(pretrained=True)
            elif mtype == 'resnet50':
                resnet = models.resnet50(pretrained=True)
            elif mtype == 'resnet101':
                resnet = models.resnet101(pretrained=True)
            elif mtype == 'resnet152':
                resnet = models.resnet152(pretrained=True)

            feat_net = nn.Sequential(*list(resnet.children())[:-1])
            in_ftrs = resnet.fc.in_features
            classifier = nn.Linear(in_features=in_ftrs, out_features=10, bias=True)
            
            model = nn.Sequential()
            model.add_module('feat_net', feat_net)
            model.add_module('flatten', nn.Flatten())
            model.add_module('classifier', classifier)
            
            self.gradcam_layer = feat_net[-2][-1]
    
        parameters = list(model.classifier.parameters())
        if not fix_ftrs:
            parameters.extend(list(model.feat_net.parameters()))
        else:
            for param in model.feat_net.parameters():
                param.requires_grad = False
            pass

        self.model = model.to(device)
        self.parameters = parameters

    def init_loaders(self):
        trainset, testset = [RIVAL10(train=train, return_masks=self.fg_only) for train in [True, False]]
        self.loaders = dict({phase:DataLoader(dset, batch_size=32, shuffle=True) 
                             for phase, dset in zip(['train', 'test'], [trainset, testset])})

    def save_model(self):
        self.model.eval()
        save_dict = dict({'state': self.model.state_dict(),
                          'acc': self.best_acc})
        torch.save(save_dict, self.save_path)
        print('\nSaved model with accuracy: {:.3f} to {}\n'.format(self.best_acc, self.save_path))

    def restore_model(self):
        print('Loading model from {}'.format(self.save_path))
        save_dict = torch.load(self.save_path)
        self.model.load_state_dict(save_dict['state'])
        self.model.eval()
        self.best_acc = save_dict['acc']

    def gradcam_layer(self):
        return self.gradcam_layer

    def process_epoch(self, phase):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
        correct, running_loss, total = 0, 0, 0
        for dat in tqdm(self.loaders[phase]):
            dat = [d.cuda() for d in dat]
            if self.fg_only:
                x, masks, y = dat
                gray = 0.5 * torch.ones(x.shape).cuda()
                x = x * masks + gray * (1-masks)
            else:
                x,y = dat
            self.optimizer.zero_grad()

            logits = self.model(x)
            loss = self.criterion(logits, y)
            if phase == 'train':
                loss.backward()
                self.optimizer.step()
            y_pred = logits.argmax(dim=1)
            correct += (y_pred == y).sum()
            total += x.shape[0]
            running_loss += loss.item()
        avg_loss, avg_acc = [stat/total for stat in [running_loss, correct]]
        return avg_loss, avg_acc
    
    def finetune(self):
        print('Beginning finetuning of model to be saved at {}'.format(self.save_path))
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self.process_epoch('train')
            test_loss, test_acc = self.process_epoch('test')
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.save_model()
            print('Epoch: {}/{}......Train Loss: {:.3f}......Train Acc: {:.3f}......Test Loss: {:.3f}......Test Acc: {:.3f}'
                  .format(epoch, self.num_epochs, train_loss, train_acc, test_loss, test_acc))
        print('Finetuning Complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RIVAL10 Finetuner')
    parser.add_argument('--mtype', type=str, default='resnet50')
    parser.add_argument('--fg_only', action="store_true")
    parser.add_argument('--all', action="store_true")

    _ALL_MTYPES = ['resnet18', 'resnet50', 'resnet101','resnet152', 'robust_resnet18', 'robust_resnet50', 
               'vit_tiny', 'vit_small', 'vit_base', 'deit_tiny', 'deit_small', 'deit_base', 
               'simclr', 'clip_RN50', 'clip_RN101', 'clip_ViT-B16', 'clip_ViT-B32', 
               'robust_resnet18_eps1', 'robust_resnet50_eps1', 'vit_small32', 'vit_base32']

    args = parser.parse_args()
    if args.all:
        for mtype in tqdm(_ALL_MTYPES):
            finetuner = FineTuner(mtype=mtype, fg_only=False, epochs=10)
            finetuner.finetune()
            finetuner = FineTuner(mtype=mtype, fg_only=True, epochs=20)
            finetuner.finetune()
    else:
        finetuner = FineTuner(mtype=args.mtype, fg_only=args.fg_only, epochs=10)
        finetuner.finetune()

