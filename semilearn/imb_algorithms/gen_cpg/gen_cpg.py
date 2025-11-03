# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import torch.nn.functional as F
from torchvision import transforms
from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.core.utils import get_data_loader
from semilearn.datasets.augmentation import RandAugment
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset
from PIL import Image
import math

@IMB_ALGORITHMS.register('gen_cpg')
class Gen_CPG(ImbAlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super(Gen_CPG, self).__init__(args, net_builder, tb_log, logger)

        #warm_up epoch
        self.warm_up = args.warm_up

        # dataset update step
        if args.dataset == 'cifar10':
            self.update_step = 5
            self.memory_step = 5
        if args.dataset == 'cifar100':
            self.update_step = 5
            self.memory_step = 5
        if args.dataset == 'food101':
            self.update_step = 5
            self.memory_step = 5
        if args.dataset == 'stl10':
            self.update_step = 5
            self.memory_step = 5

        # adaptive labeled (include the pseudo labeled) data and its dataloader
        self.current_x = None
        self.current_y = None
        self.current_idx = None
        self.current_noise_y = None
        self.current_one_hot_y = None
        self.current_one_hot_noise_y = None

        self.select_ulb_idx = None
        self.select_ulb_label = None
        self.select_ulb_pseudo_label = None
        self.select_ulb_pseudo_label_distribution = None

        self.adaptive_lb_dest = None
        self.adaptive_lb_dest_loader = None

        self.candidate_data = defaultdict(list)
        self.candidate_targets = defaultdict(list)

        self.dataset = args.dataset
        self.data = self.dataset_dict['data']
        self.targets = self.dataset_dict['targets']
        self.noised_targets = self.dataset_dict['noised_targets']
        self.lb_idx = self.dataset_dict['lb_idx']
        self.ulb_idx = self.dataset_dict['ulb_idx']

        self.mean, self.std = {}, {}

        self.mean['cifar10'] = [0.485, 0.456, 0.406]
        self.mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
        self.mean['stl10'] = [x / 255 for x in [112.4, 109.1, 98.6]]
        self.mean['svhn'] = [0.4380, 0.4440, 0.4730]
        self.mean['food101'] = [0.485, 0.456, 0.406]

        self.std['cifar10'] = [0.229, 0.224, 0.225]
        self.std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]
        self.std['stl10'] = [x / 255 for x in [68.4, 66.6, 68.5]]
        self.std['svhn'] = [0.1751, 0.1771, 0.1744]
        self.std['food101'] = [0.229, 0.224, 0.225]

        if self.dataset == 'food101':
            self.transform_weak = transforms.Compose([
                                # transforms.Resize(args.img_size),
                                transforms.RandomCrop((args.img_size, args.img_size)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(self.mean[self.dataset], self.std[self.dataset])
                                ])

            self.transform_strong = transforms.Compose([
                                # transforms.Resize(args.img_size),
                                transforms.RandomCrop((args.img_size, args.img_size)),
                                transforms.RandomHorizontalFlip(),
                                RandAugment(3, 5),
                                transforms.ToTensor(),
                                transforms.Normalize(self.mean[self.dataset], self.std[self.dataset])
                                ])
        else:
            self.transform_weak = transforms.Compose([
                                transforms.Resize(args.img_size),
                                transforms.RandomCrop(args.img_size, padding=int(args.img_size * (1 - args.crop_ratio)), padding_mode='reflect'),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(self.mean[self.dataset], self.std[self.dataset])
                                ])

            self.transform_strong = transforms.Compose([
                                transforms.Resize(args.img_size),
                                transforms.RandomCrop(args.img_size, padding=int(args.img_size * (1 - args.crop_ratio)), padding_mode='reflect'),
                                transforms.RandomHorizontalFlip(),
                                RandAugment(3, 5),
                                transforms.ToTensor(),
                                transforms.Normalize(self.mean[self.dataset], self.std[self.dataset])
                                ])

        if self.args.generated_data_dir is not None:
            gen_data, gen_targets, gen_noised_targets, self.gen_idx = self._load_generated_data(self.args.generated_data_dir)
            self.data = np.concatenate((self.data, gen_data), axis=0)
            self.targets = np.concatenate((self.targets, gen_targets), axis=0)
            self.noised_targets = np.concatenate((self.noised_targets, gen_noised_targets), axis=0)
            self.lb_idx = np.concatenate((self.lb_idx, self.gen_idx), axis=0)

            new_lb_data = self.data[self.lb_idx]
            new_lb_targets = self.targets[self.lb_idx]
            new_lb_noised_targets = self.noised_targets[self.lb_idx]
            train_lb_dataset = BasicDataset(
                img_idx=self.lb_idx,
                data=new_lb_data,
                targets=new_lb_targets,
                noised_targets=new_lb_noised_targets,
                num_classes=self.num_classes,
                is_ulb=False,
                weak_transform=self.transform_weak,
                strong_transform=self.transform_strong,
                onehot=False,
            )
        
        self.dataset_dict['train_lb'] = train_lb_dataset
        self.loader_dict['train_lb'] = get_data_loader(
            self.args,
            train_lb_dataset,
            batch_size=self.args.batch_size,
            data_sampler=self.args.train_sampler,
            num_iters=self.num_train_iter,
            num_epochs=self.epochs,
            num_workers=self.args.num_workers,
            distributed=self.distributed
        )

        # compute lb dist
        lb_class_dist = [0 for _ in range(self.num_classes)]
        if args.noise_ratio > 0:
            for c in self.dataset_dict['train_lb'].noised_targets:
                lb_class_dist[c] += 1
        else:
            for c in self.dataset_dict['train_lb'].targets:
                lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)

        self.lb_dist = torch.from_numpy(lb_class_dist.astype(np.float32)).cuda(args.gpu)

        # compute select_ulb and ulb dist
        ulb_class_dist = [0 for _ in range(self.num_classes)]
        for c in self.dataset_dict['train_ulb'].targets:
            ulb_class_dist[c] += 1
        ulb_class_dist = np.array(ulb_class_dist)

        self.ulb_dist = torch.from_numpy(ulb_class_dist.astype(np.float32)).cuda(args.gpu)

        self.select_ulb_dist = torch.zeros(self.num_classes).cuda(args.gpu)

        # compute lb_select_ulb and lb_ulb dist
        lb_ulb_class_dist = lb_class_dist + ulb_class_dist

        self.lb_ulb_dist = torch.from_numpy(lb_ulb_class_dist.astype(np.float32)).cuda(args.gpu)

        self.lb_select_ulb_dist = self.lb_dist + self.select_ulb_dist

        if self.args.candidate_pool_dir is not None:
            self._load_candidate_pool(self.args.candidate_pool_dir)

    def _load_generated_data(self, data_dir):
        class_to_idx = np.load(os.path.join(data_dir, 'class_to_idx.npy'), allow_pickle=True).item()
        generated_dir = os.path.join(data_dir, self.args.dataset)

        crop_size = self.args.img_size
        crop_ratio = self.args.crop_ratio

        gen_data = []
        gen_targets = []

        for class_name in os.listdir(generated_dir):
            class_dir = os.path.join(generated_dir, class_name)
            class_idx = class_to_idx[class_name]

            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                gen_data.append(np.array(Image.open(img_path).convert('RGB').resize((int(math.floor(crop_size / crop_ratio)), int(math.floor(crop_size / crop_ratio))))))
                gen_targets.append(class_idx)

        gen_data = np.array(gen_data)
        gen_targets = np.array(gen_targets)
        gen_noise_targets = gen_targets
        max_existing_idx = max(np.max(self.lb_idx), np.max(self.ulb_idx))

        gen_idx = np.arange(max_existing_idx + 1, max_existing_idx + 1 + len(gen_data))

        return gen_data, gen_targets, gen_noise_targets, gen_idx
    
    def _load_candidate_pool(self, data_dir):
        class_to_idx = np.load(os.path.join(data_dir, 'class_to_idx.npy'), allow_pickle=True).item()
        if self.dataset == 'food101':
            candidate_pool_dir = os.path.join(data_dir, 'food101_pool')

        crop_size = self.args.img_size
        crop_ratio = self.args.crop_ratio

        for class_name in os.listdir(candidate_pool_dir):
            class_dir = os.path.join(candidate_pool_dir, class_name)
            class_idx = class_to_idx[class_name]

            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.candidate_data[class_idx].append(np.array(Image.open(img_path).convert('RGB').resize((int(math.floor(crop_size / crop_ratio)), int(math.floor(crop_size / crop_ratio))))))
                self.candidate_targets[class_idx].append(class_idx)

    def train(self):
        """
        train function
        """
        self.model.train()
        self.call_hook("before_run")

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch

            # ~self.warm_up ce loss only and not select unlabeled data
            if self.epoch < self.warm_up:
                self.adaptive_lb_dest_loader = self.loader_dict['train_lb']
                self.lb_select_ulb_dist = self.lb_dist
                self.select_ulb_dist = torch.ones(self.num_classes).cuda(self.args.gpu)

            # self.warm_up+1~ use labeled (include the pseudo labeled) data and continue select unlabeled data
            # update the labeled (include the pseudo labeled) dataset and labeled (include the pseudo labeled) data distribution and selected unlabeled data distribution
            else:
                if self.epoch % self.memory_step == 0:
                    self.current_x = None
                    self.current_y = None
                    self.current_idx = None
                    self.current_noise_y = None
                    self.current_one_hot_y = None
                    self.current_one_hot_noise_y = None
                    self.select_ulb_pseudo_label_distribution = None

                    # process selected condident unlabeled data
                    # delete the same idx / same data contribution to gradient once
                    select_ulb_idx_to_label = {}
                    select_ulb_idx_to_pseudo_label = {}

                    for ulb_idx, ulb_pseudo_label, ulb_label in zip(self.select_ulb_idx, self.select_ulb_pseudo_label, self.select_ulb_label):
                        if ulb_idx.item() in select_ulb_idx_to_label:
                            select_ulb_idx_to_label[ulb_idx.item()].append(ulb_label.item())
                        else:
                            select_ulb_idx_to_label[ulb_idx.item()] = [ulb_label.item()]

                        if ulb_idx.item() in select_ulb_idx_to_pseudo_label:
                            select_ulb_idx_to_pseudo_label[ulb_idx.item()].append(ulb_pseudo_label.item())
                        else:
                            select_ulb_idx_to_pseudo_label[ulb_idx.item()] = [ulb_pseudo_label.item()]

                    select_ulb_unique_idx = torch.unique(self.select_ulb_idx)

                    mean_number_of_pseudo_label = []

                    for ulb_unique_idx in select_ulb_unique_idx:
                        mean_number_of_pseudo_label.append(len(select_ulb_idx_to_label[ulb_unique_idx.item()]))

                    select_ulb_unique_label = []
                    select_ulb_unique_pseudo_label = []
                    select_ulb_unique_pseudo_label_distribution = []

                    for ulb_unique_idx in select_ulb_unique_idx:
                        ulb_unique_label = select_ulb_idx_to_label[ulb_unique_idx.item()]
                        ulb_unique_pseudo_label = select_ulb_idx_to_pseudo_label[ulb_unique_idx.item()]

                        ulb_unique_pseudo_label_distribution = torch.zeros(self.num_classes)
                        for item in ulb_unique_pseudo_label:
                            ulb_unique_pseudo_label_distribution[item] += 1.0
                        ulb_unique_pseudo_label_distribution = ulb_unique_pseudo_label_distribution / torch.sum(ulb_unique_pseudo_label_distribution)

                        # process the ground-truth label                                                        
                        select_ulb_unique_label.append(torch.tensor([ulb_unique_label[0]]))

                        # process the pseudo-label
                        if len(ulb_unique_pseudo_label) > 12:
                            most_common_label = Counter(ulb_unique_pseudo_label).most_common(1)[0][0]
                            most_common_number = Counter(ulb_unique_pseudo_label).most_common(1)[0][1]
                            if most_common_number > 0.8 * len(ulb_unique_pseudo_label):
                                select_ulb_unique_pseudo_label.append(torch.tensor([most_common_label]))
                            else:
                                select_ulb_unique_pseudo_label.append(torch.tensor([-1]))
                        else:
                            select_ulb_unique_pseudo_label.append(torch.tensor([-1]))

                        # process the pseudo-label distribution
                        select_ulb_unique_pseudo_label_distribution.append(ulb_unique_pseudo_label_distribution.unsqueeze(0))

                    select_ulb_unique_label = torch.cat(select_ulb_unique_label)
                    select_ulb_unique_pseudo_label = torch.cat(select_ulb_unique_pseudo_label)
                    select_ulb_unique_pseudo_label_distribution = torch.cat(select_ulb_unique_pseudo_label_distribution)

                    self.select_ulb_idx = torch.masked_select(select_ulb_unique_idx.cpu(), select_ulb_unique_pseudo_label != -1)
                    self.select_ulb_label = torch.masked_select(select_ulb_unique_label, select_ulb_unique_pseudo_label != -1)
                    self.select_ulb_pseudo_label = torch.masked_select(select_ulb_unique_pseudo_label, select_ulb_unique_pseudo_label != -1)
                    self.select_ulb_pseudo_label_distribution = select_ulb_unique_pseudo_label_distribution[select_ulb_unique_pseudo_label != -1]

                    self.select_ulb_dist = torch.zeros(self.num_classes).cuda(self.args.gpu)
                    for item in self.select_ulb_pseudo_label:
                        self.select_ulb_dist[int(item)] += 1

                    self.lb_select_ulb_dist = self.lb_dist + self.select_ulb_dist

                    self.print_fn('select_ulb_dist:\n' + np.array_str(np.array(self.select_ulb_dist.cpu())))
                    self.print_fn('lb_select_ulb_dist:\n' + np.array_str(np.array(self.lb_select_ulb_dist.cpu())))

                    new_gen_data = []
                    new_gen_targets = []

                    target_num = torch.max(self.lb_select_ulb_dist)

                    num_to_add_per_class = torch.relu(target_num - self.lb_select_ulb_dist).int().cpu().numpy()
                    total_sampled = 0
                    for i in range(self.num_classes):
                        num_to_sample = num_to_add_per_class[i]
                        if num_to_sample == 0 or len(self.candidate_data[i]) == 0:
                            continue
                        sampled_idx = np.random.choice(len(self.candidate_data[i]), num_to_sample, replace=True)
                        new_gen_data.extend([self.candidate_data[i][j] for j in sampled_idx])
                        new_gen_targets.extend([self.candidate_targets[i][j] for j in sampled_idx])
                        total_sampled += num_to_sample
                    if total_sampled > 0:
                        new_gen_data = np.array(new_gen_data)
                        new_gen_targets = np.array(new_gen_targets)
                        num_new_gen = len(new_gen_targets)
                        self.print_fn(f'sampled {num_new_gen} images from pool.')

                        gen_one_hot_y = np.full((num_new_gen, self.num_classes), self.args.smoothing / (self.num_classes - 1))
                        gen_one_hot_y[np.arange(num_new_gen), new_gen_targets] = 1.0 - self.args.smoothing

                        start_idx = len(self.data) + len(self.select_ulb_idx) + 1
                        new_gen_idx = np.arange(start_idx, start_idx + num_new_gen)
                    else:
                        self.print_fn('no new data sampled from pool in this step.')
                        new_gen_data = np.array([])
                        new_gen_targets = np.array([])
                        new_gen_idx = np.array([])
                        gen_one_hot_y = np.array([]).reshape(0, self.num_classes)

                    '''
                    # update the current labeled and pseudo labeled data
                    self.current_idx = np.concatenate((self.lb_idx, self.select_ulb_idx), axis=0)
                    self.current_x = self.data[self.current_idx]
                    self.current_y = np.concatenate((self.targets[self.lb_idx], self.select_ulb_pseudo_label), axis=0)
                    self.current_noise_y = np.concatenate((self.noised_targets[self.lb_idx], self.select_ulb_pseudo_label), axis=0)
                    current_one_hot_y = np.full((len(self.targets[self.lb_idx]), self.num_classes), self.args.smoothing / (self.num_classes - 1))
                    current_one_hot_y[np.arange(len(self.targets[self.lb_idx])), self.targets[self.lb_idx]] = 1.0 - self.args.smoothing
                    current_one_hot_noise_y = np.full((len(self.noised_targets[self.lb_idx]), self.num_classes), self.args.smoothing / (self.num_classes - 1))
                    current_one_hot_noise_y[np.arange(len(self.noised_targets[self.lb_idx])), self.noised_targets[self.lb_idx]] = 1.0 - self.args.smoothing
                    self.current_one_hot_y = np.concatenate((current_one_hot_y, self.select_ulb_pseudo_label_distribution), axis=0)
                    self.current_one_hot_noise_y = np.concatenate((current_one_hot_noise_y, self.select_ulb_pseudo_label_distribution), axis=0)
                    '''
                    base_lb_idx = self.lb_idx
                    base_ulb_idx = self.select_ulb_idx.cpu().numpy()

                    self.current_idx = np.concatenate((base_lb_idx, base_ulb_idx, new_gen_idx), axis=0)

                    base_x = self.data[np.concatenate((base_lb_idx, base_ulb_idx))]

                    if len(new_gen_data) > 0:
                        self.current_x = np.concatenate((base_x, new_gen_data), axis=0)
                    else:
                        self.current_x = base_x
                    
                    self.current_y = np.concatenate((self.targets[base_lb_idx], self.select_ulb_pseudo_label.cpu().numpy(), new_gen_targets), axis=0)
                    self.current_noise_y = np.concatenate((self.noised_targets[base_lb_idx], self.select_ulb_pseudo_label.cpu().numpy(), new_gen_targets), axis=0)
                    
                    current_one_hot_y = np.full((len(base_lb_idx), self.num_classes), self.args.smoothing / (self.num_classes - 1))
                    current_one_hot_y[np.arange(len(base_lb_idx)), self.targets[base_lb_idx]] = 1.0 - self.args.smoothing

                    current_one_hot_noise_y = np.full((len(base_lb_idx), self.num_classes), self.args.smoothing / (self.num_classes - 1))
                    current_one_hot_noise_y[np.arange(len(base_lb_idx)), self.noised_targets[base_lb_idx]] = 1.0 - self.args.smoothing
                    
                    pseudo_label_dist = self.select_ulb_pseudo_label_distribution.cpu().numpy()
                    self.current_one_hot_y = np.concatenate((current_one_hot_y, pseudo_label_dist, gen_one_hot_y), axis=0)
                    self.current_one_hot_noise_y = np.concatenate((current_one_hot_noise_y, pseudo_label_dist, gen_one_hot_y), axis=0)

                    self.print_fn(str(self.epoch) + ': Update the labeled data.')
                    # construct the current lb_select_ulb data and its dataloader
                    self.adaptive_lb_dest = BasicDataset(self.current_idx, self.current_x, self.current_one_hot_y, self.current_one_hot_noise_y, self.args.num_classes, False, weak_transform=self.transform_weak, strong_transform=self.transform_strong, onehot=False)
                    self.adaptive_lb_dest_loader = get_data_loader(self.args, self.adaptive_lb_dest, self.args.batch_size, data_sampler=self.args.train_sampler, num_iters=self.num_train_iter, num_epochs=self.epochs, num_workers=self.args.num_workers, distributed=self.distributed)

                    self.current_x = None
                    self.current_y = None
                    self.current_idx = None
                    self.current_noise_y = None
                    self.current_one_hot_y = None
                    self.current_one_hot_noise_y = None

                    # reset select ulb idx and its pseudo label
                    self.select_ulb_idx = None
                    self.select_ulb_label = None
                    self.select_ulb_pseudo_label = None
                    self.select_ulb_pseudo_label_distribution = None

            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break

            self.call_hook("before_train_epoch")

            for data_lb, data_ulb in zip(self.adaptive_lb_dest_loader,
                                         self.loader_dict['train_ulb']):
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")

        self.call_hook("after_run")

    def train_step(self, idx_lb, x_lb_w, x_lb_s, y_lb, y_lb_noised, idx_ulb, x_ulb_w, x_ulb_s, y_ulb):
        num_lb = y_lb.shape[0]
        if self.args.noise_ratio > 0:
            lb = y_lb_noised
        else:
            lb = y_lb

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb_w, x_lb_s, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb_w, logits_x_lb_s = outputs['logits'][:2 * num_lb].chunk(2)
                aux_logits_x_lb_w, aux_logits_x_lb_s = outputs['aux_logits'][:2 * num_lb].chunk(2)                
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][2 * num_lb:].chunk(2)
                aux_logits_x_ulb_w, aux_logits_x_ulb_s = outputs['aux_logits'][2 * num_lb:].chunk(2)                
                feats_x_lb_w, feats_x_lb_s = outputs['feat'][:2 * num_lb].chunk(2)
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][2 * num_lb:].chunk(2)
            else:
                outs_x_lb_w = self.model(x_lb_w)
                logits_x_lb_w = outs_x_lb_w['logits']
                aux_logits_x_lb_w = outs_x_lb_w['aux_logits']
                feats_x_lb_w = outs_x_lb_w['feat']
                outs_x_lb_s = self.model(x_lb_s)
                logits_x_lb_s = outs_x_lb_s['logits']
                aux_logits_x_lb_s = outs_x_lb_s['aux_logits']
                feats_x_lb_s = outs_x_lb_s['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                aux_logits_x_ulb_s = outs_x_ulb_s['aux_logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    aux_logits_x_ulb_w = outs_x_ulb_w['aux_logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb_w': feats_x_lb_w, 'x_lb_s': feats_x_lb_s, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s': feats_x_ulb_s}

            if self.epoch < self.warm_up:
                # loss for labeled data
                # compute cross entropy loss
                lb_smooth = torch.zeros(num_lb, self.num_classes).cuda(self.args.gpu)
                lb_smooth.fill_(self.args.smoothing / (self.num_classes - 1))
                lb_smooth.scatter_(1, lb.unsqueeze(1), 1.0 - self.args.smoothing)
                lb = lb_smooth

                # sup_loss = self.ce_loss(logits_x_lb_w + torch.log(self.lb_select_ulb_dist / torch.sum(self.lb_select_ulb_dist)), lb, reduction='mean')
                sup_loss = self.ce_loss(logits_x_lb_w, lb, reduction='mean')

                aux_pseudo_label_w = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", logits=self.compute_prob(aux_logits_x_ulb_w.detach()), use_hard_label=self.use_hard_label, T=self.T, softmax=False)
                aux_pseudo_label_w_smooth = torch.zeros(self.args.uratio * num_lb, self.num_classes).cuda(self.args.gpu)
                aux_pseudo_label_w_smooth.fill_(self.args.smoothing / (self.num_classes - 1))
                aux_pseudo_label_w_smooth.scatter_(1, aux_pseudo_label_w.unsqueeze(1), 1.0 - self.args.smoothing)
                # aux_loss = self.ce_loss(aux_logits_x_ulb_s, aux_pseudo_label_w_smooth, reduction='mean') + self.ce_loss(aux_logits_x_lb_w + torch.log(self.lb_select_ulb_dist / torch.sum(self.lb_select_ulb_dist)), lb, reduction='mean')
                aux_loss = self.ce_loss(aux_logits_x_ulb_s, aux_pseudo_label_w_smooth, reduction='mean') + self.ce_loss(aux_logits_x_lb_w, lb, reduction='mean')

                mask = torch.tensor([False]).cuda(self.args.gpu)
            else:
                lb_smooth = torch.zeros(num_lb, self.num_classes).cuda(self.args.gpu)
                lb_smooth.fill_(self.args.smoothing / (self.num_classes - 1))
                lb_smooth.scatter_(1, lb.unsqueeze(1), 1.0 - self.args.smoothing)
                lb = lb_smooth
                # compute cross entropy loss for labeled data
                sup_loss = self.ce_loss(logits_x_lb_w, lb, reduction='mean')

                probs_x_ulb_w = self.compute_prob((logits_x_ulb_w))
                probs_x_ulb_s = self.compute_prob((logits_x_ulb_s))

                aux_pseudo_label_w = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", logits=self.compute_prob(aux_logits_x_ulb_w.detach()), use_hard_label=self.use_hard_label, T=self.T, softmax=False)
                aux_loss = self.ce_loss(aux_logits_x_ulb_s, aux_pseudo_label_w, reduction='mean') + self.ce_loss(aux_logits_x_lb_w, lb.argmax(dim=1), reduction='mean')

                energy_w = -torch.logsumexp(logits_x_ulb_w.detach(), dim=1)
                energy_s = -torch.logsumexp(logits_x_ulb_s.detach(), dim=1)
                
                # generate unlabeled targets using pseudo label hook
                pseudo_label_w = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", logits=probs_x_ulb_w, use_hard_label=self.use_hard_label, T=self.T, softmax=False)
                pseudo_label_s = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", logits=probs_x_ulb_s, use_hard_label=self.use_hard_label, T=self.T, softmax=False)

                # calculate mask
                mask_w = probs_x_ulb_w.amax(dim=-1).ge(self.p_cutoff * (1.0 - self.args.smoothing))
                mask_s = probs_x_ulb_s.amax(dim=-1).ge(self.p_cutoff * (1.0 - self.args.smoothing))
                mask_confidence = mask_w & mask_s
                mask_w_s = pseudo_label_w == pseudo_label_s

                mask_energy = (energy_w < self.args.energy_cutoff) & (energy_s < self.args.energy_cutoff)

                mask = mask_confidence & mask_w_s & mask_energy

                # update select_ulb_idx and its pseudo_label
                if self.select_ulb_idx is not None and self.select_ulb_pseudo_label is not None and self.select_ulb_label is not None:
                    self.select_ulb_idx = torch.cat([self.select_ulb_idx, idx_ulb[mask]], dim=0)
                    self.select_ulb_label = torch.cat([self.select_ulb_label, y_ulb[mask]], dim=0)
                    self.select_ulb_pseudo_label = torch.cat([self.select_ulb_pseudo_label, pseudo_label_w[mask]], dim=0)
                else:
                    self.select_ulb_idx = idx_ulb[mask]
                    self.select_ulb_label = y_ulb[mask]
                    self.select_ulb_pseudo_label = pseudo_label_w[mask]

            total_loss = sup_loss + aux_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         aux_loss=aux_loss.item(),
                                         total_loss=total_loss.item(),
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--warm_up', int, 30),
            SSL_Argument('--alpha', float, 1.0),
            SSL_Argument('--smoothing', float, 0.1),
            SSL_Argument('--generated_data_dir', str, './data/generated'),
            SSL_Argument('--candidate_pool_dir', str, './data/generated'),
            SSL_Argument('--energy_cutoff', float, -5.0)
        ]