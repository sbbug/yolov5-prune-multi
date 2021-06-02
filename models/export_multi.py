import argparse
import sys
import time

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
import torch
import torch.nn as nn
import models
from models.experimental import attempt_load
from utils.activations import Hardswish
from utils.general import set_logging, check_img_size
from models.modal_ensemble_model_uva import ModalEnseModel
from utils.torch_utils import select_device
from torch.autograd import Variable

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_visible', nargs='+', type=str,
                        default='/home/shw/code/yolov5-new/yolov5/runs/train/exp371/weights/best.pt',
                        help='model.pt path(s)')
    parser.add_argument('--weights_lwir', nargs='+', type=str,
                        default='/home/shw/code/yolov5-new/yolov5/runs/train/exp372/weights/best.pt',
                        help='model.pt path(s)')
    parser.add_argument('--img-size', nargs='+', type=int, default=[672, 672], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()
    t = time.time()

    device = select_device('0')

    # Load PyTorch model
    model = ModalEnseModel()
    model.load_weights(opt.weights_visible, opt.weights_lwir, device)

    # model = attempt_load(opt.weights_visible, map_location=device)  # load FP32 model
    # labels = model.names

    # Checks
    # gs = int(max(model.model_visible.stride))  # grid size (max stride)
    # opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size)  # image size(1,3,320,192) iDetection
    lwir = torch.zeros(opt.batch_size, 3, *opt.img_size)
    # Update model
    # for k, m in model.named_modules():
    #     m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    #     if isinstance(m, models.common.Conv) and isinstance(m.act, nn.Hardswish):
    #         # m.act = Hardswish()  # assign activation
    #         m.act = nn.LeakyReLU(0.01)
    #     # if isinstance(m, models.yolo.Detect):
    #     #     m.forward = m.forward_export  # assign forward (optional)
    # model.model_visible.model[-1].export = True  # set Detect() layer export=True
    # model.model_lwir.model[-1].export = True
    # model.model[-1].export = True
    # img = Variable(img.to(device))
    img = img.to(device, non_blocking=True)
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    lwir = lwir.to(device, non_blocking=True)
    lwir = lwir.float()
    lwir /= 255.0  # 0 - 255 to 0.0 - 1.0

    # lwir = Variable(lwir.to(device))
    # y = model(img,augment=False)  # dry run

    model.export()
    y = model(img, lwir, augment=False)  # dry run

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights_visible.replace('.pt', '.onnx')  # filename
        torch.onnx.export(model, (img, lwir), f, verbose=True, opset_version=10, input_names=['img', 'lwir'],
                          output_names=['classes', 'boxes'] if y is None else ['img_scale_1', 'img_scale_2',
                                                                               'img_scale_3', 'lwir_scale_1',
                                                                               'lwir_scale_2', 'lwir_scale_3'])

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))
