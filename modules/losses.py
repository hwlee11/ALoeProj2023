# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn

class NLLLoss(nn.NLLLoss):
    """
    NLLLoss
    """

    def __init__(self, log_probs_ndim=2, weight=None, reduction='mean', ignore_index=-100):
        """
        Args:
            log_probs_ndim (int): number of dimensions (or rank) of the logprobs tensor
            weight (list): list of rescaling weight given to each class
            reduction (str): type of the reduction over the batch
            ignore_index (int): mask out loss computation where labels = ignore_index
        """
        if weight is not None and not torch.is_tensor(weight):
            weight = torch.FloatTensor(weight)
        super().__init__(weight=weight, reduction=reduction, ignore_index=ignore_index)
        self._log_probs_dim = log_probs_ndim

    #@typecheck()
    def forward(self, log_probs, labels, loss_mask=None):
        """
        Args:
            log_probs (float): output log probability tensor
            labels (long): ground truth labels
            loss_mask (bool/float/int): tensor to specify the masking
        """
        log_probs_flatten = torch.flatten(log_probs, start_dim=0, end_dim=-2)
        labels_flatten = torch.flatten(labels, start_dim=0, end_dim=-1)

        if loss_mask is not None:
            if loss_mask.dtype is not torch.bool:
                loss_mask = loss_mask > 0.5
            loss_mask_flatten = torch.flatten(loss_mask, start_dim=0, end_dim=-1)
            log_probs_flatten = log_probs_flatten[loss_mask_flatten]
            labels_flatten = labels_flatten[loss_mask_flatten]

        if len(labels_flatten) == 0:
            return super().forward(log_probs, torch.argmax(log_probs, dim=-1))

        loss = super().forward(log_probs_flatten, labels_flatten)
        return loss

if __name__ == '__main__':
    logp = torch.rand(2,8,10)
    label = torch.tensor([[1,2,3,4,5,6,7,8],[8,7,6,5,4,3,3,1]])
    criterion = NLLLoss()

    loss = criterion(logp,label)
    print(loss)

