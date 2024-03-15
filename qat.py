import sys
import os

import yaml
import argparse
import json
from copy import deepcopy
from pathlib import Path
import warnings

# PyTorch
import torch
import torch.nn as nn

import val as validate 
from models.yolo import Model
from models.common import Conv
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download

from models.yolo import Detect, DDetect, DualDetect, DualDDetect, DetectionModel, SegmentationModel
import models.quantize as quantize

from utils.general import (LOGGER, check_dataset, check_requirements, check_img_size, colorstr, init_seeds,increment_path,file_size)
from utils.torch_utils import (torch_distributed_zero_first)


warnings.filterwarnings("ignore")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = None


class ReportTool:
    def __init__(self, file):
        self.file = file
        if os.path.exists(self.file):
            open(self.file, 'w').close()
        self.data = []

    def load_data(self):
        try:
            return json.load(open(self.file, "r"))
        except FileNotFoundError:
            return []

    def append(self, item):
        self.data.append(item)
        self.save_data()

    def update(self, item):
        for i, data_item in enumerate(self.data):
            if data_item[0] == item[0]:
                self.data[i] = item
                break
        else:
            # Se não encontrar, adiciona como um novo item
            self.append(item)
        self.save_data()

    def save_data(self):
        json.dump(self.data, open(self.file, "w"), indent=4)


def load_model(weights, device) -> Model:
    with torch_distributed_zero_first(LOCAL_RANK):
        attempt_download(weights)
    model = torch.load(weights, map_location=device)["model"]
    for m in model.modules():
        if type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    model.float()
    model.eval()
    with torch.no_grad():
        model.fuse()
    return model



def create_train_dataloader(train_path, imgsz, batch_size, single_cls, stride, hyp_path):
    with open(hyp_path) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    loader = create_dataloader(
        train_path, 
        imgsz=imgsz, 
        batch_size=batch_size, 
        single_cls=single_cls,
        augment=True, hyp=hyp, rect=False, cache=False, stride=stride, pad=0.0, image_weights=False)[0]
    return loader



def create_val_dataloader(test_path, imgsz, batch_size, single_cls, stride, keep_images=None):
    loader = create_dataloader(
        test_path, 
        imgsz=imgsz, 
        batch_size=batch_size, 
        single_cls=single_cls,
        augment=False, hyp=None, rect=True, cache=False,stride=stride,pad=0.5, image_weights=False)[0]

    def subclass_len(self):
        if keep_images is not None:
            return keep_images
        return len(self.img_files)
    
    loader.dataset.__len__ = subclass_len
    return loader

def evaluate_dataset(model_eval, val_loader, imgsz, data_dict, single_cls, save_dir, is_coco, conf_thres=0.001 , iou_thres=0.7 ):
    return validate.run(data_dict,
                        model=model_eval,
                        imgsz=imgsz,
                        single_cls=single_cls,
                        half=True,
                        task='val',
                        verbose=True,
                        conf_thres=conf_thres, 
                        iou_thres=iou_thres,
                        save_dir=save_dir,
                        save_json=is_coco,
                        dataloader=val_loader,  
                        )[0][:4]


def export_onnx(model, file, im, opset=12, dynamic=False, prefix=colorstr('QAT ONNX:')):
    check_requirements('onnx')
    import onnx

    file = Path(file)
    LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__}...')

    f = file.with_suffix('.onnx')
    output_names = ['output0', 'output1'] if isinstance(model, SegmentationModel) else ['output0']
    model.eval()
    for k, m in model.named_modules():
        # print(m)
        if isinstance(m, (Detect, DDetect, DualDetect, DualDDetect)):
            m.inplace = True
            m.dynamic = dynamic
            m.export = True
    dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
    if isinstance(model, SegmentationModel):
        dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
        dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
    elif isinstance(model, DetectionModel):
        dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
    
    quantize.export_onnx(model, im, file, opset_version=13, 
         input_names=["images"], output_names=output_names, 
         dynamic_axes=dynamic or None
     )
    
    for k, m in model.named_modules():
        if isinstance(m, (Detect, DDetect, DualDetect, DualDDetect)):
            m.inplace = True
            m.dynamic = False
            m.export = False
    

def run_quantize(weights, data, imgsz, batch_size, hyp, device, no_last_layer, save_dir, supervision_stride, iters, no_eval_origin, no_eval_ptq, eval_pycocotools, prefix=colorstr('QAT:')):
    
    if not Path(weights).exists():
        LOGGER.info(f'{prefix} Weight file not found "{weights}"  ❌')
        exit(1)
        
    quantize.initialize()
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = check_dataset(data)

    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)   # make dir

    is_coco = isinstance(data_dict.get('val'), str) and data_dict['val'].endswith(f'val2017.txt')  # COCO dataset
    if is_coco and not eval_pycocotools:
        is_coco=False
    
    nc = int(data_dict['nc'])  # number of classes
    single_cls = False if nc > 1 else True
    names = data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, data_dict)  # check
    train_path = data_dict['train']
    test_path = data_dict['val']

    result_eval_origin=None
    result_eval_ptq=None
    result_eval_qat_best=None

    

    device  = torch.device(device)
    model   = load_model(weights, device)

    if not isinstance(model, DetectionModel):
            model_name=model.__class__.__name__
            LOGGER.info(f'{prefix} {model_name} model is not supported. Only DetectionModel is supported.  ❌')
            exit(1)

    stride = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # conf onnx export
    exp_imgsz=[imgsz,imgsz]
    gs = int(max(model.stride))  # grid size (max stride)
    exp_imgsz = [check_img_size(x, gs) for x in exp_imgsz]  # verify img_size are gs-multiples
    im = torch.zeros(batch_size, 3, *exp_imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection


    train_dataloader = create_train_dataloader(train_path, imgsz, batch_size, single_cls, stride, hyp)
    val_dataloader   = create_val_dataloader(test_path, imgsz, batch_size, single_cls, stride)
    
    ### This rule is disabled - This allow user disable qat per Layers ###
    # This rule has been disabled, but it remains in the code to maintain compatibility or future implementation.
    """
    ignore_layer=-1
    if ignore_layer > -1:
        ignore_policy=f"model\.{ignore_layer}\.cv\d+\.\d+\.\d+(\.conv)?"
    else:
        ignore_policy=f"model\.9999999999\.cv\d+\.\d+\.\d+(\.conv)?"   
    """ 
    ### End ####### 
        
    quantize.replace_custom_module_forward(model)
    quantize.replace_to_quantization_module(model, ignore_policy="disabled")  ## disabled because was not implemented 
    quantize.apply_custom_rules_to_quantizer(model, lambda model, file: export_onnx(model, file, im))
    quantize.calibrate_model(model, train_dataloader, device)

    report_file = os.path.join(save_dir, "report.json")
    report = ReportTool(report_file)

    if no_eval_origin:
        LOGGER.info(f'\n{prefix} Evaluating Origin...')
        model_eval = deepcopy(model).eval()  
        with quantize.disable_quantization(model_eval):
            result_eval_origin = evaluate_dataset(model_eval, val_dataloader, imgsz, data_dict, single_cls, save_dir, is_coco )
            eval_mp, eval_mr, eval_map50, eval_map= tuple(round(x, 4) for x in result_eval_origin)
            LOGGER.info(f'\n{prefix} Eval Origin  - AP: {eval_map} AP50: {eval_map50} Precision: {eval_mp} Recall: {eval_mr}')
            report.append(["Origin", weights, eval_map, eval_map50,eval_mp, eval_mr  ])

    if no_eval_ptq:
        
        LOGGER.info(f'\n{prefix} Evaluating PTQ...')
        model_eval = deepcopy(model).eval()  
        
        if no_last_layer:
            last_layer_index = len(model_eval.model) - 1
            last_layer = model_eval.model[last_layer_index]
            if quantize.have_quantizer(last_layer):
                LOGGER.info(f'{prefix} Quantization disabled for Last Layer model.{last_layer_index}')
                quantize.disable_quantization(last_layer).apply()
        
        result_eval_ptq = evaluate_dataset(model_eval, val_dataloader, imgsz, data_dict, single_cls, save_dir, is_coco )
        eval_mp, eval_mr, eval_map50, eval_map= tuple(round(x, 4) for x in result_eval_ptq)
        LOGGER.info(f'\n{prefix} Eval PTQ - AP: {eval_map} AP50: {eval_map50} Precision: {eval_mp} Recall: {eval_mr}')
        ptq_weights = w /  f'ptq_ap_{eval_map}_{os.path.basename(weights)}'
        torch.save({"model": model_eval},f'{ptq_weights}')
        LOGGER.info(f'\n{prefix} PTQ, weights saved as {ptq_weights} ({file_size(ptq_weights):.1f} MB)')
        report.append(["PTQ", str(ptq_weights), eval_map, eval_map50,eval_mp, eval_mr ])

    best_map = 0

    def per_epoch(model, epoch, lr):
        nonlocal best_map , result_eval_qat_best

        epoch +=1
        model_eval = deepcopy(model).eval()  
        with torch.no_grad():  
            eval_result = evaluate_dataset(model_eval, val_dataloader, imgsz, data_dict, single_cls, save_dir, is_coco )
            eval_mp, eval_mr, eval_map50, eval_map= tuple(round(x, 4) for x in eval_result)
        qat_weights = w /  f'qat_ep_{epoch}_ap_{eval_map}_{os.path.basename(weights)}'
        torch.save({"model": model_eval},f'{qat_weights}')
        LOGGER.info(f'\n{prefix} Epoch-{epoch}, weights saved as {qat_weights} ({file_size(qat_weights):.1f} MB)')
        report.append([f"QAT-{epoch}", str(qat_weights), eval_map, eval_map50,eval_mp, eval_mr ])

        if eval_map > best_map:
            best_map = eval_map
            result_eval_qat_best=eval_result
            qat_weights = w /  f'qat_best_{os.path.basename(weights)}'
            torch.save({"model": model_eval}, f'{qat_weights}')
            LOGGER.info(f'{prefix} QAT Best, weights saved as {qat_weights} ({file_size(qat_weights):.1f} MB)')
            report.update(["QAT-Best", str(qat_weights), eval_map, eval_map50,eval_mp, eval_mr ])

        eval_results = [result_eval_origin, result_eval_ptq, result_eval_qat_best]
         
        LOGGER.info(f'\n\nEval Model | {"AP":<8} | {"AP50":<8} | {"Precision":<10} | {"Recall":<8}')
        LOGGER.info('-' * 55)  
        for idx, eval_r in enumerate(eval_results):
            if eval_r is not None:
                eval_mp, eval_mr, eval_map50, eval_map = tuple(round(x, 4) for x in eval_r)
                if idx == 0:
                    LOGGER.info(f'Origin     | {eval_map:<8} | {eval_map50:<8} | {eval_mp:<10} | {eval_mr:<8}')
                if idx == 1:
                    LOGGER.info(f'PTQ        | {eval_map:<8} | {eval_map50:<8} | {eval_mp:<10} | {eval_mr:<8}')
                if idx == 2:
                    LOGGER.info(f'QAT - Best | {eval_map:<8} | {eval_map50:<8} | {eval_mp:<10} | {eval_mr:<8}\n')
            
        eval_mp, eval_mr, eval_map50, eval_map= tuple(round(x, 4) for x in eval_result)
        LOGGER.info(f'\n{prefix} Eval - Epoch {epoch} | AP: {eval_map}  | AP50: {eval_map50} | Precision: {eval_mp} | Recall: {eval_mr}\n')

    def preprocess(datas):
        return datas[0].to(device).float() / 255.0

    def supervision_policy():
        supervision_list = []
        for item in model.model:
            supervision_list.append(id(item))

        keep_idx = list(range(0, len(model.model) - 1, supervision_stride))
        keep_idx.append(len(model.model) - 2)
        def impl(name, module):
            if id(module) not in supervision_list: return False
            idx = supervision_list.index(id(module))
            if idx in keep_idx:
                print(f"Supervision: {name} will compute loss with origin model during QAT training")
            else:
                print(f"Supervision: {name} no compute loss during QAT training, that is unsupervised only and doesn't mean don't learn")
            return idx in keep_idx
        return impl

    quantize.finetune(
        model, train_dataloader, no_last_layer, per_epoch, early_exit_batchs_per_epoch=iters, 
        preprocess=preprocess, supervision_policy=supervision_policy())

def run_sensitive_analysis(weights, device, data, imgsz, batch_size, hyp, save_dir, num_image, prefix=colorstr('QAT ANALYSIS:')):
    quantize.initialize()
    if not Path(weights).exists():
        LOGGER.info(f'{prefix} Weight file not found "{weights}"  ❌')
        exit(1)
        
    save_dir = Path(save_dir)
    # Create the directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=opt.exist_ok)

    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = check_dataset(data)

    is_coco=False
    
    nc = int(data_dict['nc'])  # number of classes
    single_cls = False if nc > 1 else True
    names = data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, data_dict)  # check
    train_path = data_dict['train']
    test_path = data_dict['val']

    device  = torch.device(device)
    model   = load_model(weights, device)

    if not isinstance(model, DetectionModel):
        model_name=model.__class__.__name__
        LOGGER.info(f'{prefix} {model_name} model is not supported. Only DetectionModel is supported.  ❌')
        exit(1)

    is_model_qat=False
    for i in range(0, len(model.model)):
        layer = model.model[i]
        if quantize.have_quantizer(layer):
            is_model_qat=True
            break

    if is_model_qat:
        LOGGER.info(f'{prefix} This model already quantized. Only not quantized models is allowed. ❌')
        exit(1)

    stride = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=stride)  # check image size


    train_dataloader = create_train_dataloader(train_path, imgsz, batch_size, single_cls, stride, hyp)
    val_dataloader   = create_val_dataloader(test_path, imgsz, batch_size, single_cls, stride)

    quantize.replace_to_quantization_module(model)
    quantize.calibrate_model(model, train_dataloader, device)

    report_file=os.path.join(save_dir , "summary-sensitive-analysis.json")
    report = ReportTool(report_file)

    model_eval = deepcopy(model).eval() 
    LOGGER.info(f'\n{prefix} Evaluating PTQ...')

    eval_result = evaluate_dataset(model_eval, val_dataloader, imgsz, data_dict, single_cls, save_dir, is_coco )
    eval_mp, eval_mr, eval_map50, eval_map= tuple(round(x, 4) for x in eval_result)

    LOGGER.info(f'\n{prefix} Eval PTQ - QAT enabled on All Layers - AP: {eval_map} AP50: {eval_map50} Precision: {eval_mp} Recall: {eval_mr}')
    report.append([eval_map, "PTQ"])
    LOGGER.info(f'{prefix} Sensitive analysis by each layer. Layers Detected: {len(model.model)}')

    for i in range(0, len(model.model)):
        layer = model.model[i]
        if quantize.have_quantizer(layer):
            LOGGER.info(f'{prefix} QAT disabled on Layer model.{i}')
            quantize.disable_quantization(layer).apply()
            model_eval = deepcopy(model).eval()   
            eval_result = evaluate_dataset(model_eval, val_dataloader, imgsz, data_dict, single_cls, save_dir, is_coco )
            eval_mp, eval_mr, eval_map50, eval_map= tuple(round(x, 4) for x in eval_result)
            LOGGER.info(f'\n{prefix} Eval PTQ - QAT disabled on Layer model.{i} - AP: {eval_map} AP50: {eval_map50} Precision: {eval_mp} Recall: {eval_mr}\n')
            report.append([eval_map, f"model.{i}"]) 
            quantize.enable_quantization(layer).apply()
        else:
            LOGGER.info(f'{prefix} Ignored Layer model.{i} because it is {type(layer)}')
    
    report = sorted(report.data, key=lambda x:x[0], reverse=True)
    print("Sensitive summary:")
    for n, (ap, name) in enumerate(report[:10]):
        print(f"Top{n}: Using fp16 {name}, ap = {ap:.5f}")


def run_eval(weights, device, data, imgsz, batch_size, save_dir, conf_thres, iou_thres, eval_pycocotools, prefix=colorstr('QAT TEST:')):
    
    if not Path(weights).exists():
        LOGGER.info(f'{prefix} Weight file not found "{weights}"  ❌')
        exit(1)
    
    quantize.initialize()

    save_dir = Path(save_dir)
    # Create the directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=opt.exist_ok)

    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = check_dataset(data)

   
    device  = torch.device(device)
    model   = load_model(weights, device)

    if not isinstance(model, DetectionModel):
        model_name=model.__class__.__name__
        LOGGER.info(f'{prefix} {model_name} model is not supported. Only DetectionModel is supported.  ❌')
        exit(1)

    is_model_qat=False
    for i in range(0, len(model.model)):
        layer = model.model[i]
        if quantize.have_quantizer(layer):
            is_model_qat=True
            break
    
    if not is_model_qat:
        LOGGER.info(f'{prefix} This model was not Quantized. ❌')
        exit(1)
    
    is_coco = isinstance(data_dict.get('val'), str) and data_dict['val'].endswith(f'val2017.txt')  # COCO dataset
    if is_coco and not eval_pycocotools:
        is_coco=False

    stride = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    nc = int(data_dict['nc'])  # number of classes
    single_cls = False if nc > 1 else True
    names = data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, data_dict)  # check

    test_path = data_dict['val']
    val_dataloader   = create_val_dataloader(test_path, imgsz, batch_size, single_cls, stride)

    
    LOGGER.info(f'\n{prefix} Evaluating ...')
    model_eval = deepcopy(model).eval()  
 
    result_eval = evaluate_dataset(model_eval, val_dataloader, imgsz, data_dict, single_cls, save_dir, is_coco, conf_thres=conf_thres, iou_thres=iou_thres  )
    eval_mp, eval_mr, eval_map50, eval_map= tuple(round(x, 4) for x in result_eval)
    LOGGER.info(f'\n{prefix} Eval Result - AP: {eval_map} AP50: {eval_map50} Precision: {eval_mp} Recall: {eval_mr}')
    LOGGER.info(f'\n{prefix} Eval Result, saved on {save_dir}')
   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='qat.py')
    subps  = parser.add_subparsers(dest="cmd")
    qat = subps.add_parser("quantize", help="PTQ/QAT finetune ...")

    qat.add_argument('--weights', type=str, default=ROOT / 'runs/models_original/yolov9-c.pt', help='weights path')
    qat.add_argument('--data', type=str, default=ROOT / 'data/coco.yaml', help='dataset.yaml path')
    qat.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-high.yaml', help='hyperparameters path')
    qat.add_argument("--device", type=str, default="cuda:0", help="device")
    qat.add_argument('--batch-size', type=int, default=10, help='total batch size')
    qat.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    qat.add_argument('--project', default=ROOT / 'runs/qat', help='save to project/name')
    qat.add_argument('--name', default='exp', help='save to project/name')
    qat.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    qat.add_argument("--iters", type=int, default=200, help="iters per epoch")
    qat.add_argument('--seed', type=int, default=57, help='Global training seed')
    qat.add_argument("--no-last-layer", action="store_true", help="Disable QAT on Last Layer to improve mAP but also increase Latency")
    qat.add_argument("--supervision-stride", type=int, default=1, help="supervision stride")
    qat.add_argument("--no-eval-origin", action="store_false", help="Disable eval for origin model")
    qat.add_argument("--no-eval-ptq", action="store_false", help="Disable eval for ptq model")
    qat.add_argument("--eval-pycocotools", action="store_true", help="Evalution using Pycocotools. Valid only for COCO Dataset")

    sensitive = subps.add_parser("sensitive", help="Sensitive layer analysis")
    sensitive.add_argument('--weights', type=str, default=ROOT / 'runs/models_original/yolov9-c.pt', help='Weights path (.pt)')
    sensitive.add_argument("--device", type=str, default="cuda:0", help="device")
    sensitive.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    sensitive.add_argument('--batch-size', type=int, default=10, help='total batch size')
    sensitive.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    sensitive.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-high.yaml', help='hyperparameters path') 
    sensitive.add_argument('--project', default=ROOT / 'runs/qat_sentive', help='save to project/name')
    sensitive.add_argument('--name', default='exp', help='save to project/name')
    sensitive.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    sensitive.add_argument("--num-image", type=int, default=None, help="number of image to evaluate")

    testcmd = subps.add_parser("eval", help="Do evaluate")
    testcmd.add_argument('--weights', type=str, default=ROOT / 'runs/models_original/yolov9-c.pt', help='Weights path (.pt)')
    testcmd.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    testcmd.add_argument('--batch-size', type=int, default=10, help='total batch size')
    testcmd.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='val image size (pixels)')
    testcmd.add_argument("--device", type=str, default="cuda:0", help="device")
    testcmd.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    testcmd.add_argument("--iou-thres", type=float, default=0.7, help="nms threshold")
    testcmd.add_argument('--project', default=ROOT / 'runs/qat_eval', help='save to project/name')
    testcmd.add_argument('--name', default='exp', help='save to project/name')
    testcmd.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    testcmd.add_argument("--use-pycocotools", action="store_true", help="Generate COCO annotation json format for the custom dataset")


    opt = parser.parse_args()
    if opt.cmd == "quantize":
        print(opt)
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
        init_seeds(opt.seed + 1 + RANK, deterministic=False)

        run_quantize(
            opt.weights, opt.data, opt.imgsz, opt.batch_size, 
            opt.hyp, opt.device, opt.no_last_layer, Path(opt.save_dir), 
             opt.supervision_stride, opt.iters,
            opt.no_eval_origin, opt.no_eval_ptq, opt.eval_pycocotools
        )

    elif opt.cmd == "sensitive":
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
        print(opt)
        run_sensitive_analysis(opt.weights, opt.device, opt.data, 
                               opt.imgsz, opt.batch_size, opt.hyp, 
                               opt.save_dir, opt.num_image 
                               )
    elif opt.cmd == "eval":
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
        print(opt)
        run_eval(opt.weights, opt.device, opt.data, 
                 opt.imgsz, opt.batch_size, opt.save_dir, 
                 opt.conf_thres, opt.iou_thres, opt.use_pycocotools
                 )
    else:
        parser.print_help()