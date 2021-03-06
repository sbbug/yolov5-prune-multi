'''
面向后处理的模态集成模型定义
模态主要是可将光模型与红外模型的集成
'''
import torch
import torch.nn as nn
from models.experimental import attempt_load
from utils.general import check_img_size
from models.experimental import Ensemble
import collections
from models.aware_model import AwareModel
from torchvision.transforms import Resize
from utils.general import non_max_suppression
from models.funcs import nms


class ModalEnseModel(nn.Module):

    def __init__(self, aware=False):
        super(ModalEnseModel, self).__init__()

        self.two_step = False
        self.model_visible = None
        self.model_lwir = None
        self.model_aware = None
        self.export_flag = False
        self.model_aware_path = "/home/shw/code/yolov5-master/runs/aware_model/aware.pth"
        self.torch_resize = Resize((128, 128))
        self.aware = aware

        self.conf_thres = 0.25
        self.iou_thres = 0.45
        # self.classes =
        # self.agnostic_nms =
        # self.eval()

    def forward(self, x_visible, x_lwir, img_aware, augment=False):
        fea_out_visible = None
        fea_out_lwir = None
        if self.export_flag is False:
            inf_out_visible, train_out_visible = self.model_visible(x_visible,
                                                                    augment=augment)  # inference and training outputs
            inf_out_lwir, train_out_lwir = self.model_lwir(x_lwir, augment=augment)
            # print(inf_out_visible.shape)
            # 开启光照感知
            if self.aware:
                aware_score = self.model_aware(img_aware)
                for b_ix, s in enumerate(aware_score[..., 0]):
                    inf_out_visible[b_ix][:, 5:] = inf_out_visible[b_ix][:, 5:] * s

            # 开二阶段
            if self.two_step:
                inf_out_visible = non_max_suppression(inf_out_visible, self.conf_thres, self.iou_thres)
                inf_out_lwir = non_max_suppression(inf_out_lwir, self.conf_thres, self.iou_thres)
                batch = []

                for idx, (vis, lw) in enumerate(zip(inf_out_visible, inf_out_lwir)):
                    inp = torch.cat((vis, lw), 0)
                    res, count = nms(inp[:, :4], inp[:, 4:5].squeeze(1), 0.6)
                    res = res[:count]
                    rs = torch.index_select(inp, 0, res)
                    batch.append(rs)
                return batch
            else:
                return torch.cat((inf_out_visible, inf_out_lwir), 1)
        else:
            fea_out_visible = self.model_visible(x_visible, augment=augment)  # inference and training outputs
            fea_out_lwir = self.model_lwir(x_lwir, augment=augment)
        return fea_out_visible, fea_out_lwir

    def load_weights(self, visible_model_path, lwir_model_path, device):
        if self.model_visible is None:
            self.model_visible = attempt_load(visible_model_path, map_location=device)  # load FP32 model
        if self.model_lwir is None:
            self.model_lwir = attempt_load(lwir_model_path, map_location=device)
        if self.model_aware is None and self.aware:
            self.model_aware = AwareModel()
            # print(torch.load(self.model_aware_path))
            self.model_aware.load_state_dict(torch.load(self.model_aware_path))
            self.model_aware.to(device).eval()

    def half(self):
        if self.model_visible is not None:
            self.model_visible.half()
        if self.model_lwir is not None:
            self.model_lwir.half()

    def eval_model(self):
        self.eval()
        self.model_visible.eval()
        self.model_lwir.eval()

    def check_img_shape(self, imgsz):

        imgsz = check_img_size(imgsz, s=self.model_visible.stride.max())  # check img_size
        imgsz = check_img_size(imgsz, s=self.model_lwir.stride.max())

        return imgsz

    def float(self):
        self.model_visible.float()
        self.model_lwir.float()

    def export(self):
        self.export_flag = True
        self.model_visible.model[-1].export = True
        self.model_lwir.model[-1].export = True
