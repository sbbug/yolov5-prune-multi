import argparse
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

from datasets.multi_modal_uva_datasets import create_dataloader, kaist_idx_cls
from utils.general import check_dataset, check_file, box_iou, \
    non_max_suppression, scale_coords, xywh2xyxy, set_logging, increment_path
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target
from utils.torch_utils import select_device, time_synchronized,model_info_multi_no_aware
from models.modal_ensemble_model_uva import ModalEnseModel
# from models.modal_ensemble_model_uva_aware import ModalEnseModel
from utils.torch_utils import model_info_multi

def test(data,
         weights_visible=None,
         weights_lwir=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         log_imgs=0):  # number of logged images

    set_logging()
    device = select_device(opt.device, batch_size=batch_size)

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model

    model = ModalEnseModel()
    model.load_weights(weights_visible, weights_lwir, device)
    # model = model.to(device)
    imgsz = model.check_img_shape(imgsz)
    # ema = ModelEMA(model)

    # half = device.type != 'cpu'  # half precision only supported on CUDA
    # if half:
    # model_visible.half()
    # model_lwir.half()
    # model.half()

    model_info_multi_no_aware(model,verbose=False)

    model.eval()

    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs, wandb = min(log_imgs, 100), None  # ceil
    try:
        import wandb  # Weights & Biases
    except ImportError:
        log_imgs = 0

    # Dataloader
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # _ = model.model_visible(img.half() if half else img) if device.type != 'cpu' else None  # run once
    # _ = model.model_lwir(img.half() if half else img) if device.type != 'cpu' else None  # run once
    # _ = model_visible(img.half() if half else img) if device.type != 'cpu' else None  # run once
    # _ = model_lwir(img.half() if half else img) if device.type != 'cpu' else None  # run once
    path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images
    dataloader = \
        create_dataloader(path, imgsz, batch_size, model.model_visible.stride.max(), opt, augment=False, pad=0.5,
                          rect=True)[
            0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in
             enumerate(model.model_visible.names if hasattr(model.model_visible,
                                                            'names') else model.model_visible.module.names)}

    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    # flag = False
    # y = None
    print(len(dataloader))
    sample_len = len(dataloader)
    n_s = 0
    for batch_i, (img, lwir, img_aware, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        n_s += 1
        # if flag:
        #     break
        # flag = True
        img = img.to(device, non_blocking=True)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        lwir = lwir.to(device, non_blocking=True)
        lwir = lwir.float()
        lwir /= 255.0  # 0 - 255 to 0.0 - 1.0

        img_aware = img_aware.to(device, non_blocking=True)
        img_aware = img_aware.float()
        img_aware /= 255.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            # print(img.shape, lwir.shape)
            inf_out = model(img, lwir, augment=augment)
            t0 += time_synchronized() - t

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb)
            # output = inf_out
            # print(output)
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            # shapes = (h0, w0), ((h / h0, w / w0), pad)
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))/gn).view(-1).tolist()  # normalized xywh
                    xyxy = (torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                    # line = (kaist_idx_cls[int(cls)],conf, *xyxy) # label format
                    line = kaist_idx_cls[int(cls)] + " " + str(conf) + " " + str(int(xyxy[0])) + " " + str(
                        int(xyxy[1])) + " " + str(int(xyxy[2])) + " " + str(int(xyxy[3])) + "\n"
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(line)

            # W&B logging
            if plots and len(wandb_images) < log_imgs:
                box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                             "class_id": int(cls),
                             "box_caption": "%s %.3f" % (names[cls], conf),
                             "scores": {"class_score": conf},
                             "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                wandb_images.append(wandb.Image(img[si], boxes=boxes, caption=path.name))

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(pred, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(output), paths, f, names), daemon=True).start()
    print("参与计算的样本数量", n_s * batch_size)
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple

    print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)
    print((sample_len*batch_size / t0), (sample_len*batch_size / t1), (sample_len*batch_size / (t0 + t1)))
    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb and wandb.run:
            wandb.log({"Images": wandb_images})
            wandb.log({"Validation": [wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]})

    ckpt = {'epoch': 0,
            'best_fitness': 0,
            'model': model.state_dict(),
            }

    # Save last, best and delete
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    ense = wdir / 'ense.pt'
    torch.save(ckpt, ense)

    # Return results
    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    print(f"Results saved to {save_dir}{s}")
    model.float()
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights_visible', nargs='+', type=str,
                        default='runs/train/exp88/weights/best.pt',
                        help='model.pt path(s)')
    parser.add_argument('--weights_lwir', nargs='+', type=str,
                        default='runs/train/exp89/weights/best.pt',
                        help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/uva.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=8, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=672, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', default=False, help='save results to *.txt')
    parser.add_argument('--aware', action='store_true', default=False, help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.data = check_file(opt.data)  # check file
    print(opt)

    test(opt.data,
         opt.weights_visible,
         opt.weights_lwir,
         opt.batch_size,
         opt.img_size,
         opt.conf_thres,
         opt.iou_thres,
         opt.single_cls,
         opt.augment,
         opt.verbose,
         save_txt=opt.save_txt | opt.save_hybrid,
         save_hybrid=opt.save_hybrid,
         save_conf=opt.save_conf,
         )

'''
双模态单独训练最好的效果是:
visible:/home/shw/code/yolov5-master/runs/train/exp638/weights/best.pt
lwir:/home/shw/code/yolov5-master/runs/train/exp636/weights/best.pt
'''
