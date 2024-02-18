import argparse
import json
import os
import sys
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch.nn.functional as F
import torchvision.transforms as transforms
from pycocotools import mask as maskUtils
from models.common import DetectMultiBackend
from models.yolo import SegmentationModel
from utils.callbacks import Callbacks
from utils.coco_utils import getCocoIds, getMappingId, getMappingIndex
from utils.general import (LOGGER, NUM_THREADS, TQDM_BAR_FORMAT, Profile, check_dataset, check_img_size,
                           check_requirements, check_yaml, coco80_to_coco91_class, colorstr, increment_path,
                           non_max_suppression, print_args, scale_boxes, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, box_iou
from utils.plots import output_to_target, plot_val_study
from utils.panoptic.dataloaders import create_dataloader
from utils.panoptic.general import mask_iou, process_mask, process_mask_upsample, scale_image
from utils.panoptic.metrics import Metrics, ap_per_class_box_and_mask, Semantic_Metrics
from utils.panoptic.plots import plot_images_and_masks
from utils.torch_utils import de_parallel, select_device, smart_inference_mode


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map, pred_masks):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    from pycocotools.mask import encode

    def single_encode(x):
        rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    pred_masks = np.transpose(pred_masks, (2, 0, 1))
    with ThreadPool(NUM_THREADS) as pool:
        rles = pool.map(single_encode, pred_masks)
    for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5),
            'segmentation': rles[i]})


def process_batch(detections, labels, iouv, pred_masks=None, gt_masks=None, overlap=False, masks=False):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    if masks:
        if overlap:
            nl = len(labels)
            index = torch.arange(nl, device=gt_masks.device).view(nl, 1, 1) + 1
            gt_masks = gt_masks.repeat(nl, 1, 1)  # shape(1,640,640) -> (n,640,640)
            gt_masks = torch.where(gt_masks == index, 1.0, 0.0)
        if gt_masks.shape[1:] != pred_masks.shape[1:]:
            gt_masks = F.interpolate(gt_masks[None], pred_masks.shape[1:], mode="bilinear", align_corners=False)[0]
            gt_masks = gt_masks.gt_(0.5)
        iou = mask_iou(gt_masks.view(gt_masks.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1))
    else:  # boxes
        iou = box_iou(labels[:, 1:], detections[:, :4])

    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@smart_inference_mode()
def run(
        data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        max_det=300,  # maximum detections per image
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val-pan',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        overlap=False,
        mask_downsample_ratio=1,
        compute_loss=None,
        callbacks=Callbacks(),
):
    if save_json:
        check_requirements(['pycocotools'])
        process = process_mask_upsample  # more accurate
    else:
        process = process_mask  # faster

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
        nm = de_parallel(model).model[-1].nm  # number of masks
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        nm = de_parallel(model).model.model[-1].nm if isinstance(model, SegmentationModel) else 32  # number of masks
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != 'cpu'
    #is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'coco{os.sep}val2017.txt')  # COCO dataset
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    stuff_names = data.get('stuff_names', [])  # names of stuff classes
    stuff_nc = len(stuff_names)  # number of stuff classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Semantic Segmentation
    img_id_list = []

    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, f'{weights} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
                              f'classes). Pass correct combination of --weights and --data that are trained together.'
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad, rect = (0.0, False) if task == 'speed' else (0.5, pt)  # square inference for benchmarks
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '),
                                       overlap_mask=overlap,
                                       mask_downsample_ratio=mask_downsample_ratio)[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, 'names') else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%22s' + '%11s' * 12) % ('Class', 'Images', 'Instances', 'Box(P', "R", "mAP50", "mAP50-95)", "Mask(P", "R",
                                  "mAP50", "mAP50-95)", 'S(MIoU', 'FWIoU)')
    dt = Profile(), Profile(), Profile()
    metrics = Metrics()
    semantic_metrics = Semantic_Metrics(nc = (nc + stuff_nc), device = device)
    loss = torch.zeros(6, device=device)
    jdict, stats = [], []
    semantic_jdict = []
    # callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    for batch_i, (im, targets, paths, shapes, masks, semasks) in enumerate(pbar):
        # callbacks.run('on_val_batch_start')
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
                masks = masks.to(device)
                semasks = semasks.to(device)
            masks = masks.float()
            semasks = semasks.float()
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        with dt[1]:
            preds, train_out = model(im)# if compute_loss else (*model(im, augment=augment)[:2], None)
            #train_out, preds, protos = p if len(p) == 3 else p[1]
            #preds = p
            #train_out = p[1][0] if len(p[1]) == 3 else p[0]
            # protos = train_out[-1]
            #print(preds.shape)
            #print(train_out[0].shape)
            #print(train_out[1].shape)
            #print(train_out[2].shape)
            _, pred_masks, protos, psemasks = train_out

        # Loss
        if compute_loss:
            loss += compute_loss(train_out, targets, masks, semasks = semasks)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        with dt[2]:
            preds = non_max_suppression(preds,
                                        conf_thres,
                                        iou_thres,
                                        labels=lb,
                                        multi_label=True,
                                        agnostic=single_cls,
                                        max_det=max_det,
                                        nm=nm)

        # Metrics
        plot_masks = []  # masks for plotting
        plot_semasks = []  # masks for plotting

        if training:
            semantic_metrics.update(psemasks, semasks)
        else:
            _, _, smh, smw = semasks.shape
            semantic_metrics.update(torch.nn.functional.interpolate(psemasks, size = (smh, smw), mode = 'bilinear', align_corners = False), semasks)

        if plots and batch_i < 3:
            plot_semasks.append(psemasks.clone().detach().cpu())

        for si, (pred, proto, psemask) in enumerate(zip(preds, protos, psemasks)):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            image_id = path.stem
            img_id_list.append(image_id)
            correct_masks = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            correct_bboxes = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct_masks, correct_bboxes, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
            else:
                # Masks
                midx = [si] if overlap else targets[:, 0] == si
                gt_masks = masks[midx]
                pred_masks = process(proto, pred[:, 6:], pred[:, :4], shape=im[si].shape[1:])

                # Predictions
                if single_cls:
                    pred[:, 5] = 0
                predn = pred.clone()
                scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

                # Evaluate
                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                    scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                    correct_bboxes = process_batch(predn, labelsn, iouv)
                    correct_masks = process_batch(predn, labelsn, iouv, pred_masks, gt_masks, overlap=overlap, masks=True)
                    if plots:
                        confusion_matrix.process_batch(predn, labelsn)
                stats.append((correct_masks, correct_bboxes, pred[:, 4], pred[:, 5], labels[:, 0]))  # (conf, pcls, tcls)

                pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)
                if plots and batch_i < 3:
                    plot_masks.append(pred_masks[:15].cpu())  # filter top 15 to plot

                # Save/log
                if save_txt:
                    save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
                if save_json:
                    pred_masks = scale_image(im[si].shape[1:],
                                            pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(), shape, shapes[si][1])
                    save_one_json(predn, jdict, path, class_map, pred_masks)  # append to COCO-JSON dictionary
                # callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

            # Semantic Segmentation
            h0, w0 = shape

            # resize
            _, mask_h, mask_w = psemask.shape
            h_ratio = mask_h / h0
            w_ratio = mask_w / w0

            if h_ratio == w_ratio:
                psemask = torch.nn.functional.interpolate(psemask[None, :], size = (h0, w0), mode = 'bilinear', align_corners = False)
            else:
                transform = transforms.CenterCrop((h0, w0))

                if (1 != h_ratio) and (1 != w_ratio):
                    h_new = h0 if (h_ratio < w_ratio) else int(mask_h / w_ratio)
                    w_new = w0 if (h_ratio > w_ratio) else int(mask_w / h_ratio)
                    psemask = torch.nn.functional.interpolate(psemask[None, :], size = (h_new, w_new), mode = 'bilinear', align_corners = False)

                psemask = transform(psemask)

            psemask = torch.squeeze(psemask)

            nc, h, w = psemask.shape

            semantic_mask = torch.flatten(psemask, start_dim = 1).permute(1, 0) # class x h x w -> (h x w) x class

            max_idx = semantic_mask.argmax(1)
            output_masks = torch.zeros(semantic_mask.shape).scatter(1, max_idx.cpu().unsqueeze(1), 1.0) # one hot: (h x w) x class
            output_masks = torch.reshape(output_masks.permute(1, 0), (nc, h, w)) # (h x w) x class -> class x h x w
            psemask = output_masks.to(device = device)

            # TODO: check is_coco
            instances_ids = getCocoIds(name = 'instances')
            stuff_mask = torch.zeros((h, w), device = device)
            check_semantic_mask = False
            for idx, pred_semantic_mask in enumerate(psemask):
                category_id = int(getMappingId(idx))
                if 183 == category_id:
                    # set all non-stuff pixels to other
                    pred_semantic_mask = (torch.logical_xor(stuff_mask, torch.ones((h, w), device = device))).int()

                # ignore the classes which all zeros / unlabeled class
                if (0 >= torch.max(pred_semantic_mask)) or (0 >= category_id):
                    continue

                if category_id not in instances_ids:
                    # record all stuff mask
                    stuff_mask = torch.logical_or(stuff_mask, pred_semantic_mask)

                if (category_id not in instances_ids):
                    rle = maskUtils.encode(np.asfortranarray(pred_semantic_mask.cpu(), dtype = np.uint8))
                    rle['counts'] = rle['counts'].decode('utf-8')

                    temp_d = {
                        'image_id': int(image_id) if image_id.isnumeric() else image_id,
                        'category_id': category_id,
                        'segmentation': rle,
                        'score': 1
                    }

                    semantic_jdict.append(temp_d)
                    check_semantic_mask = True

            if not check_semantic_mask:
                # append a other mask for evaluation if the image without any mask
                other_mask = (torch.ones((h, w), device = device)).int()

                rle = maskUtils.encode(np.asfortranarray(other_mask.cpu(), dtype = np.uint8))
                rle['counts'] = rle['counts'].decode('utf-8')

                temp_d = {
                    'image_id': int(image_id) if image_id.isnumeric() else image_id,
                    'category_id': 183,
                    'segmentation': rle,
                    'score': 1
                }

                semantic_jdict.append(temp_d)

        # Plot images
        if plots and batch_i < 3:
            if len(plot_masks):
                plot_masks = torch.cat(plot_masks, dim=0)
            if len(plot_semasks):
                plot_semasks = torch.cat(plot_semasks, dim = 0)
            plot_images_and_masks(im, targets, masks, semasks, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)
            plot_images_and_masks(im, output_to_target(preds, max_det=15), plot_masks, plot_semasks, paths,
                                  save_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred

        # callbacks.run('on_val_batch_end')

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        results = ap_per_class_box_and_mask(*stats, plot=plots, save_dir=save_dir, names=names)
        metrics.update(results)
    nt = np.bincount(stats[4].astype(int), minlength=nc)  # number of targets per class

    # Print results
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 10  # print format
    LOGGER.info(pf % ("all", seen, nt.sum(), *metrics.mean_results(), *semantic_metrics.results()))
    if nt.sum() == 0:
        LOGGER.warning(f'WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels')

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(metrics.ap_class_index):
            LOGGER.info(pf % (names[c], seen, nt[c], *metrics.class_result(i), *semantic_metrics.results()))

    # Print speeds
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
    # callbacks.run('on_val_end')

    mp_bbox, mr_bbox, map50_bbox, map_bbox, mp_mask, mr_mask, map50_mask, map_mask = metrics.mean_results()
    miou_sem, fwiou_sem = semantic_metrics.results()
    semantic_metrics.reset()

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_path = Path(data.get('path', '../coco'))
        anno_json = str(anno_path / 'annotations/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        semantic_anno_json = str(anno_path / 'annotations/stuff_val2017.json')  # annotations json
        semantic_pred_json = str(save_dir / f"{w}_predictions_stuff.json")  # predictions json
        LOGGER.info(f'\nsaving {semantic_pred_json}...')
        with open(semantic_pred_json, 'w') as f:
            json.dump(semantic_jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            results = []
            for eval in COCOeval(anno, pred, 'bbox'), COCOeval(anno, pred, 'segm'):
                if is_coco:
                    eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # img ID to evaluate
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                results.extend(eval.stats[:2])  # update results (mAP@0.5:0.95, mAP@0.5)
            map_bbox, map50_bbox, map_mask, map50_mask = results

            # Semantic Segmentation
            from utils.stuff_seg.cocostuffeval import COCOStuffeval

            LOGGER.info(f'\nEvaluating pycocotools stuff... ')
            imgIds = [int(x) for x in img_id_list]

            stuffGt = COCO(semantic_anno_json)  # initialize COCO ground truth api
            stuffDt = stuffGt.loadRes(semantic_pred_json)  # initialize COCO pred api

            cocoStuffEval = COCOStuffeval(stuffGt, stuffDt)
            cocoStuffEval.params.imgIds = imgIds  # image IDs to evaluate
            cocoStuffEval.evaluate()
            stats, statsClass = cocoStuffEval.summarize()
            stuffIds = getCocoIds(name = 'stuff')
            title = ' {:<5} | {:^6} | {:^6} '.format('class', 'iou', 'macc') if (0 >= len(stuff_names)) else \
                    ' {:<5} | {:<20} | {:^6} | {:^6} '.format('class', 'class name', 'iou', 'macc')
            print(title)
            for idx, (iou, macc) in enumerate(zip(statsClass['ious'], statsClass['maccs'])):
                id = (idx + 1)
                if id not in stuffIds:
                    continue
                content = ' {:<5} | {:0.4f} | {:0.4f} '.format(str(id), iou, macc) if (0 >= len(stuff_names)) else \
                            ' {:<5} | {:<20} | {:0.4f} | {:0.4f} '.format(str(id), str(stuff_names[getMappingIndex(id, name = 'stuff')]), iou, macc)
                print(content)

        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    final_metric = mp_bbox, mr_bbox, map50_bbox, map_bbox, mp_mask, mr_mask, map50_mask, map_mask, miou_sem, fwiou_sem
    return (*final_metric, *(loss.cpu() / len(dataloader)).tolist()), metrics.get_maps(nc), t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128-pan.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo-pan.pt', help='model path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val-pan', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    # opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    #check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/
            LOGGER.warning(f'WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results')
        if opt.save_hybrid:
            LOGGER.warning('WARNING ⚠️ --save-hybrid returns high mAP from hybrid labels, not from predictions alone')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != 'cpu'  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolo.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolo.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
