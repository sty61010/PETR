import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import numpy as np


class GridMask(nn.Module):
    def __init__(self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch  # + 1.#0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.rand() > self.prob or not self.training:
            return x
        n, c, h, w = x.size()
        x = x.view(-1, h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, h)
        step_length = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        # The first dimension is necessary for F.rotate, which accepts a (C, H, W) image tensor.
        mask = x.new_ones((1, hh, ww), dtype=torch.int)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            hh_range = torch.arange(hh)
            shifted_hh_range = (hh_range % d - st_h) % d
            # Set all pixels whose index are between [`st_h``, `st_h` + `step_length`) after modded by `d`
            mask[0, ((shifted_hh_range >= 0) & (shifted_hh_range < step_length) &
                     (hh_range >= st_h) & (hh_range < min((hh // d - 1) * d + st_h + step_length, hh)))] = 0
        if self.use_w:
            ww_range = torch.arange(ww)
            shifted_ww_range = (ww_range % d - st_w) % d
            # Set all pixels whose index are between [`st_h``, `st_h` + `step_length`) after modded by `d`
            mask[0, :, ((shifted_ww_range >= 0) & (shifted_ww_range < step_length) &
                        (ww_range >= st_w) & (ww_range < min((ww // d - 1) * d + st_w + step_length, ww)))] = 0

        r = np.random.randint(self.rotate)
        mask = F.rotate(mask, r)
        mask = mask[..., (hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        if self.mode == 1:
            mask = 1 - mask
        mask = mask.expand_as(x)
        if self.offset:
            # random sample from [-1, 1]
            offset = 2 * (torch.rand(h, w, device=x.device) - 0.5)
            x = torch.where(mask, x, offset)
        else:
            x *= mask

        return x.view(n, c, h, w)
