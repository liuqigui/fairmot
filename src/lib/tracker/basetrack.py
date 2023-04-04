from matplotlib import pyplot as plt
import numpy as np
from collections import OrderedDict
import logging
import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import time
import math
import torchvision


from .common import *
from .experimental import *
from .autoanchor import check_anchor_order
from .nms_pytorch import soft_nms_pytorch,cluster_nms,cluster_SPM_nms,cluster_diounms,cluster_SPM_dist_nms


try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None

def set_logging(rank=-1, verbose=True):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if (verbose and rank in [-1, 0]) else logging.WARN)

logger = logging.getLogger(__name__)

class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


def parse_model(d, ch):  # model_dict, input_channels(3)

    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    out_list = []
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args

        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain

        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3, C3TR]:

            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            if i == 24:
                args = [c2, c1, *args[1:]]

            else:
                args = [c1, c2, *args[1:]]

            if m in [BottleneckCSP, C3, C3TR]:
                args.insert(2, n)  # number of repeats
                n = 1

        elif m is nn.BatchNorm2d:
            args = [ch[f]]

        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
            # c2 = sum([ch[x] for x in f])

        # elif m is Contract:
        #     c2 = ch[f] * args[0] ** 2
        # elif m is Expand:
        #     c2 = ch[f] // args[0] ** 2

        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            out_list += [i]

        elif m is SAAN:
            out_list += [i]
            args.append([ch[x + 1] for x in f])
        elif m is DenseMask:
            out_list += [i]
            args.append([ch[x + 1] for x in f])

        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module

        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        # if i == 0:
        #     ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save), sorted(out_list)


class DenseMask(nn.Module):
    def __init__(self, mask=1, ch=()):
        super(DenseMask, self).__init__()
        self.proj1 = Conv(ch[0] // 2, 1, k=3)
        self.proj2 = nn.ConvTranspose2d(ch[1], 1, 4, stride=2,
                                        padding=1, output_padding=0,
                                        groups=1, bias=False)
        self.proj3 = nn.ConvTranspose2d(ch[2], 1, 8, stride=4,
                                        padding=2, output_padding=0,
                                        groups=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, layers):
        return self.sigmoid(self.proj1(layers[0][0]) + self.proj2(layers[1][0]) + self.proj3(layers[2][0]))


class SAAN(nn.Module):
    def __init__(self, id_embedding=256, ch=()):
        super(SAAN, self).__init__()
        self.proj1 = nn.Sequential(Conv(ch[0] // 2, 256, k=3),
                                   SAAN_Attention(k_size=3, ch=256, s_state=True, c_state=False))
        self.proj2 = nn.Sequential(Conv(ch[1], 256, k=3),
                                   nn.ConvTranspose2d(256, 256, 4, stride=2,
                                                      padding=1, output_padding=0,
                                                      groups=256, bias=False),
                                   SAAN_Attention(k_size=3, ch=256, s_state=True, c_state=False))
        self.proj3 = nn.Sequential(Conv(ch[2], 256, k=3),
                                   nn.ConvTranspose2d(256, 256, 8, stride=4,
                                                      padding=2, output_padding=0,
                                                      groups=256, bias=False),
                                   SAAN_Attention(k_size=3, ch=256, s_state=True, c_state=False))

        self.node = nn.Sequential(SAAN_Attention(k_size=3, ch=256 * 3, s_state=False, c_state=True),
                                  Conv(256 * 3, 256, k=3),
                                  nn.Conv2d(256, 512,
                                            kernel_size=1, stride=1,
                                            padding=0, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, layers):
        layers[0] = self.proj1(layers[0][1])
        layers[1] = self.proj2(layers[1][1])
        layers[2] = self.proj3(layers[2][1])
        # layers[0] = self.proj1(layers[0])
        # layers[1] = self.proj2(layers[1])
        # layers[2] = self.proj3(layers[2])
        id_layer_out = self.node(torch.cat([layers[0], layers[1], layers[2]], 1))
        id_layer_out = id_layer_out.permute(0, 2, 3, 1).contiguous()
        return id_layer_out


class SAAN_Attention(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size=3, ch=256, s_state=False, c_state=False):
        super(SAAN_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()
        # self.conv1 = Conv(ch, ch,k=1)

        self.s_state = s_state
        self.c_state = c_state

        if c_state:
            self.c_attention = nn.Sequential(nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False),
                                             nn.LayerNorm([1, ch]),
                                             nn.LeakyReLU(0.3, inplace=True),
                                             nn.Linear(ch, ch, bias=False))

        if s_state:
            self.conv_s = nn.Sequential(Conv(ch, ch // 4, k=1))
            self.s_attention = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # channel_attention
        if self.c_state:
            y_avg = self.avg_pool(x)
            y_max = self.max_pool(x)
            y_c = self.c_attention(y_avg.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1) + \
                  self.c_attention(y_max.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            y_c = self.sigmoid(y_c)

        # spatial_attention
        if self.s_state:
            x_s = self.conv_s(x)
            avg_out = torch.mean(x_s, dim=1, keepdim=True)
            max_out, _ = torch.max(x_s, dim=1, keepdim=True)
            y_s = torch.cat([avg_out, max_out], dim=1)
            y_s = self.sigmoid(self.s_attention(y_s))

        if self.c_state and self.s_state:
            y = x * y_s * y_c + x
        elif self.c_state:
            y = x * y_c + x
        elif self.s_state:
            y = x * y_s + x
        else:
            y = x
        return y


class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), id_embedding=256, ch=(), inplace=False):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList([nn.Conv2d(ch[0] // 2, (self.no) * self.na, 1),
                                nn.Conv2d(ch[1], (self.no) * self.na, 1),
                                nn.Conv2d(ch[2], (self.no) * self.na, 1)])  # output conv

        # self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)
        self.id_embedding = id_embedding
        self.k = Parameter(torch.ones(1) * 10)

    def forward(self, x):

        x_ori = x.copy()  # for profiling
        z = []  # inference output
        p = []
        for i in range(self.nl):
            # print(x[i][0].size())

            x[i] = self.m[i](x[i][0])  # conv
            # x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                p.append(y.clone())
                y[..., 0:2] = ((y[..., 0:2] - 0.5) * self.k + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                y = y[..., :6]
                p[-1][..., 2:] = y[..., 2:]
                z.append(y.view(bs, -1, self.no))
            else:
                if self.stride != None:
                    if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                        self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                    y = x[i].sigmoid()
                    p.append(y.clone())
                    y[..., 0:2] = ((y[..., 0:2] - 0.5) * self.k + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = y[..., :6]
                    z.append(y.view(bs, -1, self.no))
                else:
                    z = [torch.zeros((1,1,1)).to(x[i].device)]*3

        # return x if self.training else (torch.cat(z, 1), x)
        return [x,self.k,x_ori,torch.cat(z, 1)] if self.training else (torch.cat(z, 1), [x,self.k,x_ori,p,self.stride,self.grid,self.anchor_grid,self.no,bs])

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def feature_visualization(x, module_type, stage, n=32, save_dir=Path('runs/detect/exp')):
    """
    x:              Features to be visualized
    module_type:    Module type
    stage:          Module stage within model
    n:              Maximum number of feature maps to plot
    save_dir:       Directory to save results
    """
    if 'Detect' not in module_type:
        batch, channels, height, width = x.shape  # batch, channels, height, width
        if height > 1 and width > 1:
            f = f"stage{stage}_{module_type.split('.')[-1]}_features.png"  # filename

            blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
            n = min(n, channels)  # number of plots
            fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 rows x n/8 cols
            ax = ax.ravel()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            for i in range(n):
                ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
                ax[i].axis('off')

            print(f'Saving {save_dir / f}... ({n}/{channels})')
            plt.savefig(save_dir / f, dpi=300, bbox_inches='tight')

def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class
    max_det = 1000  # maximum number of detections per image

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], self.conf, iou_thres=self.iou, classes=self.classes, max_det=self.max_det)



class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict

        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model

        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # setting input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save, self.out = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)
        # logger.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 128  # 2x min stride
            # s = 256  # 2x min stride
            m.inplace = self.inplace

            x = self.forward(torch.zeros(2, ch, s, s))

            m.stride = torch.tensor([s / x.shape[-2] for x in x[2][0]])  # forward
            # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # logger.info('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self.forward_augment(x)  # augmented inference, None
        # return self.forward_once(x, profile, visualize)  # single-scale inference, train
        return self.forward_once(x, profile)  # single-scale inference, train

    def forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self.forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        return torch.cat(y, 1), None  # augmented inference, train

    def forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        output = []
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                if m == self.model[0]:
                    logger.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
                logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')

            x = m(x)  # run
            if m.i in self.out:
                output.append(x)
            y.append(x if m.i in self.save else None)  # save output

            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)

        if profile:
            logger.info('%.1fms total' % sum(dt))
        return output

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            logger.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             logger.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            logger.info('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            logger.info('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add AutoShape module
        logger.info('Adding AutoShape... ')
        m = AutoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)
 
 
def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


def model_info(model, verbose=False, img_size=640):
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPs
        from thop import profile
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), device=next(model.parameters()).device)  # input
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPs
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        fs = ', %.1f GFLOPs' % (flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPs
    except (ImportError, Exception):
        fs = ''

    logger.info(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")
 
        
class AutoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class
    max_det = 1000  # maximum number of detections per image

    def __init__(self, model):
        super(AutoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('AutoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/images/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, str):  # filename or uri
                im, f = Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im), im
                im = np.asarray(exif_transpose(im))
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(y, self.conf, iou_thres=self.iou, classes=self.classes, max_det=self.max_det)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


class recheck_Box(nn.Module):
    def __init__(self,channal_base=256):
        super(recheck_Box, self).__init__()
        self.proj1 = Conv(channal_base, channal_base,k=3)
        self.proj2 = nn.ConvTranspose2d(channal_base*2, channal_base, 4, stride=2,
                                                       padding=1, output_padding=0,
                                                       groups=channal_base, bias=False)
        self.proj3 = nn.ConvTranspose2d(channal_base*4, channal_base, 8, stride=4,
                                                      padding=2, output_padding=0,
                                                      groups=channal_base, bias=False)
        self.conv_box = nn.Sequential(Conv(channal_base, channal_base, k=3),
                                   nn.Conv2d(channal_base, 4, kernel_size=3, stride=1, padding=1, bias=True))

    def forward(self,layers):
        x_det = self.proj1(layers[0][0]) + self.proj2(layers[1][0]) + self.proj3(layers[2][0])
        x_box = self.conv_box(x_det)
        return x_box.permute(0,2,3,1)


class recheck_heatmap(nn.Module):
    def __init__(self,channal_base=256):
        super(recheck_heatmap, self).__init__()
        self.proj1 = Conv(channal_base, channal_base,k=3)
        self.proj2 = nn.ConvTranspose2d(channal_base*2, channal_base, 4, stride=2,
                                                       padding=1, output_padding=0,
                                                       groups=channal_base, bias=False)
        self.proj3 = nn.ConvTranspose2d(channal_base*4, channal_base, 8, stride=4,
                                                      padding=2, output_padding=0,
                                                      groups=channal_base, bias=False)

        self.conv1 = nn.Sequential(Conv(1, channal_base,k=3),
                                  nn.Conv2d(channal_base, 1, kernel_size=3, stride=1, padding=1, bias=True))
        self.conv2 = nn.Sequential(Conv(channal_base, channal_base,k=3),
                                   Conv(channal_base, channal_base, k=3),
                                   nn.Conv2d(channal_base, 1, kernel_size=3, stride=1, padding=1, bias=True))
        self.sigmoid = nn.Sigmoid()


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def build_targets_siammot(p, targets, model,train_state=False):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    if train_state:
        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
    else:
        det = model.model.module.model[-1] if is_parallel(model.model) else model.model.model[-1]  # Detect() module
    na, nt = det.na, targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices= [], [], []
    gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

    g = 0.5  # bias
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # offsets

    gain[2:6] = torch.tensor(p[0].shape)[[3, 2, 3, 2]]  # xyxy gain

    # Match targets to anchors
    t = targets * gain

    # Define
    b, c = t[0][:, :2].long().T  # image, class
    gxy = t[0][:, 2:4]  # grid xy
    gwh = t[0][:, 4:6]  # grid wh
    gij = gxy.long()
    gi, gj = gij.T  # grid xy indices
    gj[gj>=int(gain[3])] = int(gain[3])-1
    gj[gj < 0] = 0
    gi[gi>=int(gain[2])] = int(gain[2])-1
    gi[gi < 0] = 0

    tbox.append(torch.cat((gxy - gij, gwh), 1))
    tcls.append(c)  # class
    indices.append((b, gj, gi))  # image,grid indices
    return tcls, tbox, indices


def non_max_suppression_and_inds(prediction, conf_thres=0.1, iou_thres=0.6, dense_mask=[], merge=False, classes=None,
                                 agnostic=False, method='standard'):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = 1  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:6] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:6] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:6].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), x[:, 6:]), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        if method == 'standard':
            i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if method == 'soft':
            i = soft_nms_pytorch(boxes, scores, sigma=0.5, thresh=0.2, cuda=1)
        if method == "cluster":
            i = cluster_nms(boxes, scores, iou_thres)
        if method == "cluster_SPM":
            i = cluster_SPM_nms(boxes, scores, iou_thres)
        if method == "cluster_diou":
            i = cluster_diounms(boxes, scores, iou_thres, dense_mask)
        if method == "cluster_SPM_dist":
            i = cluster_SPM_dist_nms(boxes, scores, iou_thres)

        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass
        output[xi] = x[i]

        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    if len(x) != 0:
        x_inds = (output[0][:, 0] + output[0][:, 2]) // 16
        y_inds = (output[0][:, 1] + output[0][:, 3]) // 16
        y_inds[y_inds >= 76] = 75
        # y_inds[y_inds < 0] = 0
        x_inds[x_inds >= 136] = 135
        # x_inds[x_inds < 0] = 0
        x_inds = x_inds.cpu().numpy().tolist()
        y_inds = y_inds.cpu().numpy().tolist()
        # x_inds = [int(x) for x in x_inds]
        # y_inds = [int(x) for x in y_inds]
    else:
        return [], [], []
    return output[0].cpu(), x_inds, y_inds

class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict

        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model

        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # setting input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save, self.out = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)
        # logger.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 128  # 2x min stride
            # s = 256  # 2x min stride
            m.inplace = self.inplace

            x = self.forward(torch.zeros(2, ch, s, s))

            m.stride = torch.tensor([s / x.shape[-2] for x in x[2][0]])  # forward
            # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # logger.info('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self.forward_augment(x)  # augmented inference, None
        # return self.forward_once(x, profile, visualize)  # single-scale inference, train
        return self.forward_once(x, profile)  # single-scale inference, train

    def forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self.forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        return torch.cat(y, 1), None  # augmented inference, train

    def forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        output = []
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                if m == self.model[0]:
                    logger.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
                logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')

            x = m(x)  # run
            if m.i in self.out:
                output.append(x)
            y.append(x if m.i in self.save else None)  # save output

            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)

        if profile:
            logger.info('%.1fms total' % sum(dt))
        return output

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            logger.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             logger.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            logger.info('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            logger.info('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add AutoShape module
        logger.info('Adding AutoShape... ')
        m = AutoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

class SiamMot(nn.Module):
    def __init__(self,opt,model):
        super(SiamMot, self).__init__()
        self.opt = opt
        channel_dict = {"s":128,"m":192,"l":256,"x":320}
        channal_base = channel_dict[opt.cfg.split(".")[-2][-1]]
        self.model = model
        self.stride = self.model.stride
        self.siambox = recheck_Box(channal_base=channal_base)
        self.siamheatmap = recheck_heatmap(channal_base=channal_base)
        self.r = self.opt.radius

    def forward(self,img,Tracklets_T=None,targets=None,previous_box=None):
        pred = self.model(img)
        if targets == None and Tracklets_T==None:
            return pred
        else:
            if targets != None:
                id_embeding, dense_mask, p, p_Det = pred[0], pred[1], pred[2][0], pred[2][3]
                batch_list = []
                id_F = []
                tcls, tbox, indices = build_targets_siammot(p, targets, self.model,train_state = True)
                b, gj, gi = indices[0]
                for b_i in range(len(p_Det)):
                    batch_list.append(sum(b == b_i))
                    p_Det_s = p_Det[b_i][p_Det[b_i][ :, 4] > self.opt.conf_thres]
                    dets, x_inds, y_inds = non_max_suppression_and_inds(p_Det_s.unsqueeze(0), self.opt.conf_thres, 0.5, dense_mask=dense_mask[b_i].unsqueeze(0),method='cluster_diou')
                    x_ori = gi[sum(batch_list[:len(batch_list) - 1]):sum(batch_list)]
                    y_ori = gj[sum(batch_list[:len(batch_list) - 1]):sum(batch_list)]
                    x_inds = torch.tensor(x_inds)
                    y_inds = torch.tensor(y_inds)
                    inds_add = torch.cat([x_inds.unsqueeze(0),y_inds.unsqueeze(0)],dim=0).permute(1,0).cuda()
                    inds = torch.cat([x_ori.unsqueeze(0),y_ori.unsqueeze(0)],dim=0).permute(1,0)
                    for i in range(len(inds_add)):
                        if inds_add[i] not in inds:
                            inds = torch.cat([inds,inds_add[i].unsqueeze(0)],dim=0)
                    inds = inds.long()
                    id_F.append(id_embeding[b_i][inds[:,1],inds[:,0]])

                '''
                tcls, tbox, indices = build_targets_siammot(p, targets, self.model,train_state = True)
                b, gj, gi = indices[0]
                id_f = id_embeding[indices[0]].detach()
                batch_list = []
                id_F = []
                for batch_i in range(max(b) + 1):
                    batch_list.append(sum(b == batch_i))
                    id_F.append(id_f[sum(batch_list[:len(batch_list) - 1]):sum(batch_list)])
                '''
                for batch_i in range(len(id_F) // 2):
                    id_F[batch_i * 2], id_F[batch_i * 2 + 1] = id_F[batch_i * 2 + 1], id_F[batch_i * 2]
                x_det = pred[2][2]

            if Tracklets_T != None:
                id_embeding = pred[0]
                _, train_out = pred[2]
                x_det = train_out[2]
                id_F = Tracklets_T

        for i in range(len(id_F)):
            h, w, c = id_embeding[i].size()
            k, c = id_F[i].size()
            y = torch.matmul(F.normalize(id_embeding[i].view(h * w, c), dim=1), F.normalize(id_F[i], dim=1).t()).view(h, w,k).permute(2, 0, 1)
            c, h, w = y.size()
            if Tracklets_T != None:
                y_mask = torch.zeros((c, h, w)).cuda()
            else:
                y_mask = torch.zeros((c,h,w)).cuda().half()
            y = torch.where(y > 0, y, y_mask)
            if self.opt.Global_Point:
                for j in range(len(y)):
                    y_ind = torch.nonzero(y[j] == torch.max(y[j]))[0]
                    y_mask[j][y_ind[0]-self.r:y_ind[0]+self.r, y_ind[1]-self.r:y_ind[1]+self.r] = 1
                y = y * y_mask

            y = torch.sum(y,dim=0)
            y = (y/torch.max(y)).unsqueeze(0).unsqueeze(0)
            if i == 0:
                x_F = y
            else:
                x_F = torch.cat([x_F, y], dim=0)
        if Tracklets_T == None and len(x_F) < len(id_embeding):
            x_F = torch.cat([x_F, torch.zeros((len(id_embeding)-len(x_F),1,h,w)).cuda()], dim=0)

        siambox = self.siambox(x_det)
        hmmap = self.siamheatmap(x_F,x_det)
        return [pred, hmmap, siambox]