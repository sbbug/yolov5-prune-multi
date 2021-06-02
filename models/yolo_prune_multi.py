'''
yolo.py备份

'''

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path
from torch.autograd import Variable
import math
import torch
import torch.nn as nn
import yaml
from models.SE import SE, Fus, PAM_Module
from models.aware_model import AwareModel

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.common import Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, Concat, NMS, autoShape, C3_Res_S
from models.experimental import MixConv2d, CrossConv
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info_raw, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None

ANCHORS = [
    [10, 13, 16, 30, 33, 23],  # P3/8
    [30, 61, 62, 45, 59, 119],  # P4/16
    [116, 90, 156, 198, 373, 326]
]


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):

        if self.export:
            for i in range(self.nl):
                x[i] = self.m[i](x[i])
                bx, _, ny, nx = x[i].shape
                x[i] = x[i].permute(0, 2, 3, 1).contiguous()
            return x
        else:
            # stride = torch.tensor([8, 16, 32])
            # x = x.copy()  # for profiling
            z = []  # inference output
            self.training |= self.export
            for i in range(self.nl):
                x[i] = self.m[i](x[i])  # conv
                # print(x[i].shape)
                bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,85,20,20) to x(bs,3,20,20,85)
                x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

                if not self.training:  # inference
                    if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                        self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                    y = x[i].sigmoid()
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None,
                 device=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        # whether to aware
        self.aware = False
        self.model_aware_path = "/home/shw/code/yolov5-master/runs/aware_model/aware.pth"

        self.fuse = False

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model_vs, self.save_vs = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.model_lw, self.save_lw = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # self.detect_fus = Detect(nc, ANCHORS, [128, 256, 512])  # detect for fusion feature

        self.detect = Detect(nc, ANCHORS, [128, 256, 512])  # detect for visible feature
        self.detect_l = Detect(nc, ANCHORS, [128, 256, 512])  # detect for lwir feature

        # define fusion unit
        if self.fuse:
            self.ses = {
                0: SE(128 * 2, 16),
                1: SE(256 * 2, 16),
                2: SE(512 * 2, 16)
            }
            self.fusion = {
                0: Fus(128 * 2, 128),
                1: Fus(256 * 2, 256),
                2: Fus(512 * 2, 512)
            }

        # define visible aware model
        if self.aware:
            self.model_aware = AwareModel()
            # print(torch.load(self.model_aware_path))
            self.model_aware.load_state_dict(torch.load(self.model_aware_path))

        if isinstance(self.detect, Detect):
            # visible modal
            s = 256  # 2x min stride
            self.detect.stride = torch.tensor(
                [s / x.shape[-2] for x in
                 self.forward(torch.zeros(1, ch, s, s), torch.zeros(1, ch, s, s))[0]])  # forward
            self.detect.anchors /= self.detect.stride.view(-1, 1, 1)

            ##
            self.detect_l.stride = self.detect.stride
            self.detect_l.anchors = self.detect.anchors
            self.stride = self.detect.stride

            if self.fuse:
                self.detect_fus.stride = self.detect.stride
                self.detect_fus.anchors = self.detect.anchors

            # lwir modal
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())
        if self.fuse:
            for k in self.ses.keys():
                self.ses[k].to(device)
                self.fusion[k].to(device)
        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, v_x, l_x):

        return self.forward_once(v_x, l_x)  # single-scale inference, train

    def forward_once(self, v_x, l_x):
        y_vs, dt_vs = [], []  # outputs
        y_lw, dt_lw = [], []

        vs_ts = []  # three detect tensor for visible
        lw_ts = []  # three detect tensor for lwir
        for i, (m_vs, m_lw) in enumerate(zip(self.model_vs, self.model_lw)):

            # for visible modal
            if m_vs.f != -1:  # if not from previous layer
                v_x = y_vs[m_vs.f] if isinstance(m_vs.f, int) else [v_x if j == -1 else y_vs[j] for j in
                                                                    m_vs.f]  # from earlier layers

            v_x = m_vs(v_x)  # run
            y_vs.append(v_x if m_vs.i in self.save_vs else None)  # save output

            # for lwir modal
            if m_lw.f != -1:  # if not from previous layer
                l_x = y_lw[m_lw.f] if isinstance(m_lw.f, int) else [l_x if j == -1 else y_lw[j] for j in
                                                                    m_lw.f]  # from earlier layers

            l_x = m_lw(l_x)  # run
            y_lw.append(l_x if m_lw.i in self.save_lw else None)  # save output

            if i in [17, 20, 23]:
                vs_ts.append(v_x)
                lw_ts.append(l_x)

        # fusion and detect
        det_x = []
        if self.fuse:
            for i, (vs, lw) in enumerate(zip(vs_ts, lw_ts)):
                inp = torch.cat((vs, lw), 1)
                # inp = self.PAM[i](inp)
                det_x.append(self.fusion[i](inp * self.ses[i](inp)))
            out1 = self.detect_fus(det_x)
        out2 = self.detect(vs_ts)
        out3 = self.detect_l(lw_ts)

        if self.fuse:
            return out1, out2, out3

        return out2, out3

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.detect  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.detect  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
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
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info_raw(self, verbose, img_size)

    def load_state_para(self, vis_par, lw_par):
        self.model_vs.load_state_dict(vis_par)
        self.model_lw.load_state_dict(lw_par)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3_Res_S]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3_Res_S]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()
