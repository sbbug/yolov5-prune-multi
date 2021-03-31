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


class ModalEnseModel(nn.Module):

    def __init__(self):
        super(ModalEnseModel, self).__init__()

        self.model_visible = None
        self.model_lwir = None
        self.export_flag = False
        # self.eval()

    def forward(self, x_visible, x_lwir, augment=False):

        if self.export_flag is False:
            inf_out_visible, train_out_visible = self.model_visible(x_visible,
                                                                    augment=augment)  # inference and training outputs
            inf_out_lwir, train_out_lwir = self.model_lwir(x_lwir, augment=augment)
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
