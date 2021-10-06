import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from resnet1d import resnet18_1d
from resnet3d import BasicBlock_SF, ResNet_SF

class GlobalLocal(nn.Module):
    def __init__(self):
        super(GlobalLocal, self).__init__()

        self.a_model = resnet18_1d()

        self.v_slow = ResNet_SF(
            BasicBlock_SF,
            [2,2,2,2],
            num_classes=400,
            shortcut_type='B',
            sample_size=112,
            sample_duration=16,
            sf='slow'
        )

        self.v_fast = ResNet_SF(
            BasicBlock_SF,
            [2,2,2,2],
            num_classes=400,
            shortcut_type='B',
            sample_size=112,
            sample_duration=16,
            sf='fast'
        )
        self.a_proj1 = nn.Conv1d(512, 512, kernel_size=1)
        self.a_proj2 = nn.Conv1d(512, 64, kernel_size=1)

        self.v_proj1 = nn.Conv3d(512, 512, kernel_size=1)
        self.v_proj2 = nn.Conv3d(64, 64, kernel_size=1)

        self.criterion = nn.CrossEntropyLoss()

    def extract_feat(self, img):
        return img

    def forward(self, img, audio):
        device = img.device
        img_slow = img[:, :, ::8, :, :]

        b, h, w = audio.shape
        a = audio.permute(0, 2, 1)


        a = self.a_model(a) # 32 512 49
        v_slow = self.v_slow(img_slow) # 32 512 4 8 8
        v_slow = self.v_proj1(v_slow)
        v_fast = self.v_fast(img) # 32 64 31 8 8
        v_fast = self.v_proj2(v_fast)

        # a slow loss
        a_proj1 = self.a_proj1(a)
        a_mean = a_proj1.mean(dim=2) # 32 * 512
        v_slow_mean = v_slow.mean(dim=2) # 32 *512 * 8 * 8
        b, c, h, w = v_slow_mean.shape
        v_slow_tmp = v_slow_mean.permute(1, 0, 2, 3).reshape(c, -1)


        dot = torch.mm(a_mean, v_slow_tmp) # 32,  32 *8 * 8
        dot = dot.view(b, b, -1) # 32 32 h*w

        nominator = dot * torch.eye(b)[:, :, None].to(device=dot.device)
        nominator = nominator.sum(dim=1)
        nominator = torch.logsumexp(nominator, dim=1)
        denominator = torch.cat((dot, dot.permute(1, 0, 2)), dim=1).view(b, -1)
        denominator = torch.logsumexp(denominator, dim=1)
        loss_global = torch.mean(denominator - nominator)

        ###### 

        # attention weight
        a_tmp = a_mean.view(b, 1, c) # 32 1 512
        v_slow_tmp = v_slow_mean.view(b, c, -1) # 32 512 64
        dot = torch.bmm(a_tmp, v_slow_tmp) # 32 1 64
        dot = F.softmax(dot, dim=-1)
        tmp = dot.argmax(dim=-1)
        weight = dot.view(b, h, w) # 32 8 8
        ######

        # a fast loss
        a_proj2 = self.a_proj2(a) # 32 64 98
        b, c, t = a_proj2.shape
        weight = dot.view(b, 1, 1, h, w) # 32, 1, 1, 8 8
        v_fast_tmp = v_fast * weight
        v_fast_tmp = v_fast_tmp.sum(dim=(3,4)) # 32 64 31
        _, _, t2 = v_fast_tmp.shape

        a_tmp = a_proj2.permute(0, 2, 1).reshape(-1, c)# 32*98  64
        v_fast_tmp = v_fast_tmp.permute(1, 0, 2).reshape(c, -1)# 64 32*31

        dot = torch.mm(a_tmp, v_fast_tmp) #98 *31
        clip_size = a_tmp.shape[0] // v_fast_tmp.shape[1] * v_fast_tmp.shape[1]
        dot = dot[:clip_size, :]
        dot = dot.permute(1, 0)
        b = dot.shape[0]
        dot = dot.view(b, b, -1)
        nominator = dot * torch.eye(b)[:, :, None].to(device=dot.device)
        nominator = nominator.sum(dim=1)
        nominator = torch.logsumexp(nominator, dim=1)
        denominator = torch.cat((dot, dot.permute(1, 0, 2)), dim=1).view(b, -1)
        denominator = torch.logsumexp(denominator, dim=1)
        loss_local = torch.mean(denominator - nominator)

        return loss_global, loss_local

