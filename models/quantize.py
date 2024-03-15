################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################


import os
import re
from typing import List, Callable, Union, Dict
from tqdm import tqdm
from copy import deepcopy

# PyTorch
import torch
import torch.optim as optim
from torch.cuda import amp

from utils.loss_tal import ComputeLoss

# Pytorch Quantization
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules
from pytorch_quantization import tensor_quant
from absl import logging as quant_logging

import torch.nn.functional as F

from utils.general import (LOGGER,colorstr,check_amp)
                           
# Custom Rules
from models.quantize_rules import find_quantizer_pairs
import models


### These classes has not been utilized yet; it's still undergoing testing. #####
### BEGIN #####
""" 
class QuantSiLU(torch.nn.Module, quant_nn_utils.QuantInputMixin):
    def __init__(self, **kwargs):
        super(QuantSiLU, self).__init__()
        self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
        self._input1_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
        self._input0_quantizer._calibrator._torch_hist = True
        self._input1_quantizer._calibrator._torch_hist = True
        
    def forward(self, input):
        return self._input0_quantizer(input) * self._input1_quantizer(torch.sigmoid(input))
    
class QuantRepNCSPELAN4Chunk(torch.nn.Module):
        def __init__(self, c):
            super().__init__()
            self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
            self.c = c
        def forward(self, x, chunks, dims):
            return torch.split(self._input0_quantizer(x), (self.c, self.c), dims)

class QuantUpsample(torch.nn.Module): 
        def __init__(self, size, scale_factor, mode):
            super().__init__()
            self.size = size
            self.scale_factor = scale_factor
            self.mode = mode
            self._input_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
            
        def forward(self, x):
            return F.interpolate(self._input_quantizer(x), self.size, self.scale_factor, self.mode) 
"""
           
### END #####
        
class QuantConcat(torch.nn.Module, quant_nn_utils.QuantInputMixin):
    def __init__(self, dimension=1):
        super(QuantConcat, self).__init__()
        self._input_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
        self._input_quantizer._calibrator._torch_hist = True
        self.dimension = dimension         
  
    def forward(self, inputs):
        inputs = [self._input_quantizer(input) for input in inputs]
        return torch.cat(inputs, self.dimension)
              
class QuantAdd(torch.nn.Module, quant_nn_utils.QuantMixin):
    def __init__(self, quantization):
        super().__init__()
        
        if quantization:
            self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
            self._input1_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self.quantization = quantization

    def forward(self, x, y):
        if self.quantization:
            #rint(f"QAdd {self._input0_quantizer}  {self._input1_quantizer}")
            return self._input0_quantizer(x) + self._input1_quantizer(y)
        return x + y
                
  

class disable_quantization:
    def __init__(self, model):
        self.model  = model

    def apply(self, disabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = disabled

    def __enter__(self):
        self.apply(True)

    def __exit__(self, *args, **kwargs):
        self.apply(False)


class enable_quantization:
    def __init__(self, model):
        self.model  = model

    def apply(self, enabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = not enabled

    def __enter__(self):
        self.apply(True)
        return self

    def __exit__(self, *args, **kwargs):
        self.apply(False)


def have_quantizer(module):
    for name, module in module.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            return True


# Initialize PyTorch Quantization
def initialize():
    quant_modules.initialize( )
    quant_desc_input = QuantDescriptor(calib_method="histogram")
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantAvgPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

    quant_logging.set_verbosity(quant_logging.ERROR)

    quant_modules._DEFAULT_QUANT_MAP.extend(
             [quant_modules._quant_entry(models.common, "Concat", QuantConcat)]
         )
 


def transfer_torch_to_quantization(nninstance : torch.nn.Module, quantmodule):
    quant_instance = quantmodule.__new__(quantmodule)
    for k, val in vars(nninstance).items():
        setattr(quant_instance, k, val)

    def __init__(self):
        if self.__class__.__name__ == 'QuantConcat': 
           self.__init__()
        elif isinstance(self, quant_nn_utils.QuantInputMixin):
            quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
            self.init_quantizer(quant_desc_input)

            # Turn on torch_hist to enable higher calibration speeds
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
        else:
            quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
            self.init_quantizer(quant_desc_input, quant_desc_weight)
            # Turn on torch_hist to enable higher calibration speeds
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
                self._weight_quantizer._calibrator._torch_hist = True

    __init__(quant_instance)
    return quant_instance


def quantization_ignore_match(ignore_policy : Union[str, List[str], Callable], path : str) -> bool:

    if ignore_policy is None: return False
    if isinstance(ignore_policy, Callable):
        return ignore_policy(path)

    if isinstance(ignore_policy, str) or isinstance(ignore_policy, List):

        if isinstance(ignore_policy, str):
            ignore_policy = [ignore_policy]

        if path in ignore_policy: return True
        for item in ignore_policy:
            if re.match(item, path):
                return True
    return False

 
def replace_to_quantization_module(model : torch.nn.Module, ignore_policy : Union[str, List[str], Callable] = None, prefixx=colorstr('QAT:')):

    module_dict = {}
    for entry in quant_modules._DEFAULT_QUANT_MAP:
        module = getattr(entry.orig_mod, entry.mod_name)
        module_dict[id(module)] = entry.replace_mod

    def recursive_and_replace_module(module, prefix=""):
        for name in module._modules:
            submodule = module._modules[name]
            path      = name if prefix == "" else prefix + "." + name
            recursive_and_replace_module(submodule, path)

            submodule_id = id(type(submodule))
            if submodule_id in module_dict:
                ignored = quantization_ignore_match(ignore_policy, path)
                if ignored:
                    LOGGER.info(f'{prefixx} Quantization: {path} has ignored.')
                    continue
                    
                module._modules[name] = transfer_torch_to_quantization(submodule, module_dict[submodule_id])

    recursive_and_replace_module(model)


def get_attr_with_path(m, path):
    def sub_attr(m, names):
        name = names[0]
        value = getattr(m, name)
        if len(names) == 1:
            return value
        return sub_attr(value, names[1:])
    return sub_attr(m, path.split("."))


def repnbottleneck_quant_forward(self, x):
    if hasattr(self, "repaddop"):
        return self.repaddop(x, self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))
    return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


### These def has not been utilized yet; it's still undergoing testing. #####
### Begin #### 
""" 
def concat_quant_forward(self, x):
        if hasattr(self, "concatop"):
            return self.concatop(x, self.d)
        return torch.cat(x, self.d)

def upsample_quant_forward(self, x):
        if hasattr(self, "upsampleop"):
            return self.upsampleop(x)
        return F.interpolate(x)

def repncspelan4_qaunt_forward(self, x):
        if hasattr(self, "repncspelan4chunkop"):
            y = list(self.repncspelan4chunkop(self.cv1(x),2, 1))
            y.extend(m(y[-1])  for m in [self.cv2, self.cv3])
            return self.cv4(torch.cat(y, 1))
        else:
            y = list(self.cv1(x).split((self.c, self.c), 1))
            y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
            return self.cv4(torch.cat(y, 1))
"""
### End #### 

def apply_custom_rules_to_quantizer(model : torch.nn.Module, export_onnx : Callable):

    # apply rules to graph
    export_onnx(model,  "quantization-custom-rules-temp.onnx")
    pairs = find_quantizer_pairs("quantization-custom-rules-temp.onnx")
    for major, sub in pairs:
        print(f"Rules: {sub} match to {major}")
        get_attr_with_path(model, sub)._input_quantizer = get_attr_with_path(model, major)._input_quantizer
    os.remove("quantization-custom-rules-temp.onnx")

    for name, module in model.named_modules():
        if module.__class__.__name__ == "RepNBottleneck":
            if module.add:
                print(f"Rules: {name}.add match to {name}.cv1")
                major = module.cv1.conv._input_quantizer
                module.repaddop._input0_quantizer = major
                module.repaddop._input1_quantizer = major

def replace_custom_module_forward(model):
    for name, module  in model.named_modules():
            if module.__class__.__name__ == "RepNBottleneck":
                if module.add:
                    if not hasattr(module, "repaddop"):
                        #print(f"Add QuantAdd to {name}")
                        module.repaddop = QuantAdd(module.add)
                    module.__class__.forward = repnbottleneck_quant_forward
    

def calibrate_model(model : torch.nn.Module, dataloader, device, num_batch=25):

    def compute_amax(model, **kwargs):
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax()
                    else:
                        module.load_calib_amax(**kwargs)

                    module._amax = module._amax.to(device)
        
    def collect_stats(model, data_loader, device, num_batch=200):
        """Feed data to the network and collect statistics"""
        # Enable calibrators
        model.eval()
        for name, module in model.named_modules():

            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        # Feed data to the network for collecting stats
        with torch.no_grad():
            for i, datas in tqdm(enumerate(data_loader), total=num_batch, desc="Collect stats for calibrating"):
                imgs = datas[0].to(device, non_blocking=True).float() / 255.0
                model(imgs)

                if i >= num_batch:
                    break

        # Disable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()
    
    with torch.no_grad():
        collect_stats(model, dataloader, device, num_batch=num_batch)
        compute_amax(model, method="percentile", percentile=99.999, strict=False) # strict=False avoid Exception when some quantizer are never used
   


def finetune(
    model : torch.nn.Module, train_dataloader, no_last_layer, per_epoch_callback : Callable = None, preprocess : Callable = None,
    nepochs=10, early_exit_batchs_per_epoch=1000, lrschedule : Dict = None, fp16=True, learningrate=1e-5,
    supervision_policy : Callable = None, prefix=colorstr('QAT:')
):
    origin_model = deepcopy(model).eval()
    disable_quantization(origin_model).apply()
                    
    model.train()
    model.requires_grad_(True)

    scaler       = amp.GradScaler(enabled=fp16)
    optimizer    = optim.Adam(model.parameters(), learningrate)
    quant_lossfn = torch.nn.MSELoss()
    device       = next(model.parameters()).device

    if no_last_layer:
        last_layer_index = len(model.model) - 1
        last_layer = model.model[last_layer_index]
        if have_quantizer(last_layer):
            LOGGER.info(f'{prefix} Quantization disabled for Last Layer model.{last_layer_index}')
            disable_quantization(last_layer).apply()

    if lrschedule is None:
        lrschedule = {
            0: 1e-6,
            3: 1e-5,
            8: 1e-6
        }


    def make_layer_forward_hook(l):
        def forward_hook(m, input, output):
            l.append(output)
        return forward_hook

    supervision_module_pairs = []
    for ((mname, ml), (oriname, ori)) in zip(model.named_modules(), origin_model.named_modules()):
        if isinstance(ml, quant_nn.TensorQuantizer): continue

        if supervision_policy:
            if not supervision_policy(mname, ml):
                continue

        supervision_module_pairs.append([ml, ori])


    for iepoch in range(nepochs):

        if iepoch in lrschedule:
            learningrate = lrschedule[iepoch]
            for g in optimizer.param_groups:
                g["lr"] = learningrate

        model_outputs  = []
        origin_outputs = []
        remove_handle  = []
        


        for ml, ori in supervision_module_pairs:
            remove_handle.append(ml.register_forward_hook(make_layer_forward_hook(model_outputs))) 
            remove_handle.append(ori.register_forward_hook(make_layer_forward_hook(origin_outputs)))

        model.train()
        pbar = tqdm(train_dataloader, desc="QAT", total=early_exit_batchs_per_epoch)
        for ibatch, imgs in enumerate(pbar):

            if ibatch >= early_exit_batchs_per_epoch:
                break
            
            if preprocess:
                imgs = preprocess(imgs)
                

            imgs = imgs.to(device)
            with amp.autocast(enabled=fp16):
                model(imgs)

                with torch.no_grad():
                    origin_model(imgs)

                quant_loss = 0
                for mo, fo in zip(model_outputs, origin_outputs):
                    for m, f in zip(mo, fo):
                        quant_loss += quant_lossfn(m, f)

                model_outputs.clear()
                origin_outputs.clear()

            if fp16:
                scaler.scale(quant_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                quant_loss.backward()
                optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(f"QAT Finetuning {iepoch + 1} / {nepochs}, Loss: {quant_loss.detach().item():.5f}, LR: {learningrate:g}")

        # You must remove hooks during onnx export or torch.save
        for rm in remove_handle:
            rm.remove()

        if per_epoch_callback:
            if per_epoch_callback(model, iepoch, learningrate):
                break


def export_onnx(model, input, file, *args, **kwargs):
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    model.eval()

    with torch.no_grad():
        torch.onnx.export(model, input, file, *args, **kwargs)

    quant_nn.TensorQuantizer.use_fb_fake_quant = False
