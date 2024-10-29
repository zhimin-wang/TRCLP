"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        debug = False
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0] # 256
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None: # coming here
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            if debug:
                print("labels.shape1: ", labels.shape)
            mask = torch.eq(labels, labels.T).float().to(device) # 256×256

        else:
            mask = mask.float().to(device)
        np.set_printoptions(threshold=np.inf) # 256, 
        if debug:
            print("mask.shape: ", mask.shape) # 256×256
        # print("mask: ", mask)
        contrast_count = features.shape[1] # 2
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # print("torch.unbind(features, dim=1): ", torch.unbind(features, dim=1).shape)
        if debug:
            print("contrast_feature.shape: ", contrast_feature.shape) # 512, 384
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all': # coming here
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature) # [512, 512]
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count) # [512, 512]
        if debug:
            print("mask2.shape: ", mask.shape)
        # mask-out self-contrast cases
        # Out-of-place version of torch.Tensor.scatter_() torch.scatter(input, dim, index, src) https://blog.csdn.net/guofei_fly/article/details/104308528
        logits_mask = torch.scatter( 
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        #logits_mask
        # [[0., 1., 1.,  ..., 1., 1., 1.],
        # [1., 0., 1.,  ..., 1., 1., 1.],
        # [1., 1., 0.,  ..., 1., 1., 1.],
        # ...,
        # [1., 1., 1.,  ..., 0., 1., 1.],
        # [1., 1., 1.,  ..., 1., 0., 1.],
        # [1., 1., 1.,  ..., 1., 1., 0.]], device='cuda:0'

        #mask
        # [[1., 0., 0.,  ..., 0., 0., 1.],
        # [0., 1., 0.,  ..., 1., 0., 0.],
        # [0., 0., 1.,  ..., 0., 0., 0.],
        # ...,
        # [0., 1., 0.,  ..., 1., 0., 0.],
        # [0., 0., 0.,  ..., 0., 1., 0.],
        # [1., 0., 0.,  ..., 0., 0., 1.]], device='cuda:0'

        mask = mask * logits_mask

        # [[0., 0., 0.,  ..., 0., 0., 1.],
        # [0., 0., 0.,  ..., 1., 0., 0.],
        # [0., 0., 0.,  ..., 0., 0., 0.],
        # ...,
        # [0., 1., 0.,  ..., 0., 0., 0.],
        # [0., 0., 0.,  ..., 0., 0., 0.],
        # [1., 0., 0.,  ..., 0., 0., 0.]], device='cuda:0'

        if debug:
            print("mask:", mask)

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask 

        if debug:
            print("exp_logits: ", exp_logits)
        
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        if debug:
            print("log_prob: ", log_prob)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        if debug:
            print("mean_log_prob_pos: ", mean_log_prob_pos)
        
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        loss = loss.view(anchor_count, batch_size).mean()
        if debug:
            print("loss: ", loss)
        return loss
