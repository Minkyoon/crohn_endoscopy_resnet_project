import torch
import torch.nn as nn
import numpy as np
import topk.functional as F

from topk.utils import detect_large
from topk.polynomial.sp import log_sum_exp, LogSumExp

from topk.utils import delta, split



class _SVMLoss(nn.Module):

    def __init__(self, n_classes, alpha):

        assert isinstance(n_classes, int)

        assert n_classes > 0
        assert alpha is None or alpha >= 0

        super(_SVMLoss, self).__init__()
        self.alpha = alpha if alpha is not None else 1
        self.register_buffer('labels', torch.from_numpy(np.arange(n_classes)))
        self.n_classes = n_classes
        self._tau = None

    def forward(self, x, y):

        raise NotImplementedError("Forward needs to be re-implemented for each loss")

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, tau):
        if self._tau != tau:
            print("Setting tau to {}".format(tau))
            self._tau = float(tau)
            self.get_losses()

    def cuda(self, device=None):
        nn.Module.cuda(self, device)
        self.get_losses()
        return self

    def cpu(self):
        nn.Module.cpu()
        self.get_losses()
        return self


class MaxTop1SVM(_SVMLoss):

    def __init__(self, n_classes, alpha=None):

        super(MaxTop1SVM, self).__init__(n_classes=n_classes,
                                         alpha=alpha)
        self.get_losses()

    def forward(self, x, y):
        return self.F(x, y).mean()

    def get_losses(self):
        self.F = F.Top1_Hard_SVM(self.labels, self.alpha)


class MaxTopkSVM(_SVMLoss):

    def __init__(self, n_classes, alpha=None, k=5):

        super(MaxTopkSVM, self).__init__(n_classes=n_classes,
                                         alpha=alpha)
        self.k = k
        self.get_losses()

    def forward(self, x, y):
        return self.F(x, y).mean()

    def get_losses(self):
        self.F = F.Topk_Hard_SVM(self.labels, self.k, self.alpha)

#### 오류가 뜨는 코드
# class SmoothTop1SVM(_SVMLoss):
#     def __init__(self, n_classes, alpha=None, tau=1.):
#         super(SmoothTop1SVM, self).__init__(n_classes=n_classes,
#                                             alpha=alpha)
#         self.tau = tau
#         self.thresh = 1e3
#         self.get_losses()

#     def forward(self, x, y):
#         smooth, hard = detect_large(x, 1, self.tau, self.thresh)

#         loss = 0
#         if smooth.data.sum():
#             x_s, y_s = x[smooth], y[smooth]
#             x_s = x_s.view(-1, x.size(1))
#             loss += self.F_s(x_s, y_s).sum() / x.size(0)
#         if hard.data.sum():
#             x_h, y_h = x[hard], y[hard]
#             x_h = x_h.view(-1, x.size(1))
#             loss += self.F_h(x_h, y_h).sum() / x.size(0)

#         return loss

#     def Top1_Smooth_SVM(self, labels, tau, alpha=1.):
#         def fun(x, y):
#             # add loss term and subtract ground truth score
#             x = x + delta(y, labels, alpha) - x.gather(1, y[:, None])
#             # compute loss
#             loss = tau * log_sum_exp(x / tau)
#             return loss
#         return fun

#     def Top1_Hard_SVM(self, labels, alpha=1.):
#         def fun(x, y):
#             # max oracle
#             max_, _ = (x + delta(y, labels, alpha)).max(1)
#             # subtract ground truth
#             loss = max_ - x.gather(1, y[:, None]).squeeze()
#             return loss
#         return fun

#     def get_losses(self):
#         self.F_h = self.Top1_Hard_SVM(self.labels, self.alpha)
#         self.F_s = self.Top1_Smooth_SVM(self.labels, self.tau, self.alpha)
# 오류가 뜨는 코드를 직렬화
# class SmoothTop1SVM(_SVMLoss):
#     def __init__(self, n_classes, alpha=None, tau=1.):
#         super(SmoothTop1SVM, self).__init__(n_classes=n_classes,
#                                             alpha=alpha)
#         self.tau = tau
#         self.thresh = 1e3
#         self.get_losses()

#     def forward(self, x, y):
#         smooth, hard = detect_large(x, 1, self.tau, self.thresh)

#         loss = 0
#         if smooth.data.sum():
#             x_s, y_s = x[smooth], y[smooth]
#             x_s = x_s.view(-1, x.size(1))
#             loss += self.F_s(x_s, y_s).sum() / x.size(0)
#         if hard.data.sum():
#             x_h, y_h = x[hard], y[hard]
#             x_h = x_h.view(-1, x.size(1))
#             loss += self.F_h(x_h, y_h).sum() / x.size(0)

#         return loss

#     def Top1_Smooth_SVM(self, x, y, labels, tau, alpha=1.):
#         # add loss term and subtract ground truth score
#         y = y.to(x.device)
#         labels = labels.to(x.device)
        
#         x = x + delta(y, labels, alpha) - x.gather(1, y[:, None])
#         # compute loss
#         loss = tau * log_sum_exp(x / tau)
#         return loss

#     def Top1_Hard_SVM(self, x, y, labels, alpha=1.):
#         # max oracle
#         y = y.to(x.device)
#         labels = labels.to(x.device)
#         max_, _ = (x + delta(y, labels, alpha)).max(1)
#         # subtract ground truth
#         loss = max_ - x.gather(1, y[:, None]).squeeze()
#         return loss

#     def get_losses(self):
#         self.F_h = lambda x, y: self.Top1_Hard_SVM(x, y, self.labels, self.alpha)
#         self.F_s = lambda x, y: self.Top1_Smooth_SVM(x, y, self.labels, self.tau, self.alpha)


class SmoothTop1SVM(_SVMLoss):
    def __init__(self, n_classes, alpha=None, tau=1.):
        super(SmoothTop1SVM, self).__init__(n_classes=n_classes,
                                            alpha=alpha)
        self.tau = tau
        self.thresh = 1e3
        self.get_losses()

    def forward(self, x, y):
        smooth, hard = detect_large(x, 1, self.tau, self.thresh)

        loss = 0
        if smooth.data.sum():
            x_s, y_s = x[smooth], y[smooth]
            x_s = x_s.view(-1, x.size(1))
            loss += self.calculate_F_s(x_s, y_s).sum() / x.size(0)
        if hard.data.sum():
            x_h, y_h = x[hard], y[hard]
            x_h = x_h.view(-1, x.size(1))
            loss += self.calculate_F_h(x_h, y_h).sum() / x.size(0)

        return loss
    def Top1_Smooth_SVM(self, x, y, labels, tau, alpha=1.):
        # add loss term and subtract ground truth score
        y = y.to(x.device)
        labels = labels.to(x.device)
        
        x = x + delta(y, labels, alpha) - x.gather(1, y[:, None])
        # compute loss
        loss = tau * log_sum_exp(x / tau)
        return loss

    def Top1_Hard_SVM(self, x, y, labels, alpha=1.):
        # max oracle
        y = y.to(x.device)
        labels = labels.to(x.device)
        max_, _ = (x + delta(y, labels, alpha)).max(1)
        # subtract ground truth
        loss = max_ - x.gather(1, y[:, None]).squeeze()
        return loss

    def calculate_F_s(self, x, y):
        return self.Top1_Smooth_SVM(x, y, self.labels, self.tau, self.alpha)

    def calculate_F_h(self, x, y):
        return self.Top1_Hard_SVM(x, y, self.labels, self.alpha)

    def get_losses(self):
        pass




class SmoothTopkSVM(_SVMLoss):

    def __init__(self, n_classes, alpha=None, tau=1., k=5):
        super(SmoothTopkSVM, self).__init__(n_classes=n_classes,
                                            alpha=alpha)
        self.k = k
        self.tau = tau
        self.thresh = 1e3
        self.get_losses()

    def forward(self, x, y):
        smooth, hard = detect_large(x, self.k, self.tau, self.thresh)

        loss = 0
        if smooth.data.sum():
            x_s, y_s = x[smooth], y[smooth]
            x_s = x_s.view(-1, x.size(1))
            loss += self.F_s(x_s, y_s).sum() / x.size(0)
        if hard.data.sum():
            x_h, y_h = x[hard], y[hard]
            x_h = x_h.view(-1, x.size(1))
            loss += self.F_h(x_h, y_h).sum() / x.size(0)

        return loss

    def get_losses(self):
        self.F_h = F.Topk_Hard_SVM(self.labels, self.k, self.alpha)
        self.F_s = F.Topk_Smooth_SVM(self.labels, self.k, self.tau, self.alpha)
