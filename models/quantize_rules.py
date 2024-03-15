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

import onnx

def find_with_input_node(model, name):
    for node in model.graph.node:
        if len(node.input) > 0 and name in node.input:
            return node

def find_all_with_input_node(model, name):
    all = []
    for node in model.graph.node:
        if len(node.input) > 0 and name in node.input:
            all.append(node)
    return all

def find_with_output_node(model, name):
    for node in model.graph.node:
        if len(node.output) > 0 and name in node.output:
            return node

""" 
def find_with_no_change_parent_node(model, node):
    parent = find_with_output_node(model, node.input[0])
    if parent is not None:
        print("Parent:", parent.op_type)
        if parent.op_type in ["Concat", "MaxPool", "AveragePool", "Slice"]:
            return find_with_no_change_parent_node(model, parent)
    return parent 
"""

def find_quantizelinear_conv(model, qnode):
    dq   = find_with_input_node(model, qnode.output[0])
    conv = find_with_input_node(model, dq.output[0])
    return conv


def find_quantize_conv_name(model, weight_qname):
    dq = find_with_output_node(model, weight_qname)
    q  = find_with_output_node(model, dq.input[0])
    return ".".join(q.input[0].split(".")[:-1])

def find_quantizer_pairs(onnx_file):

    model = onnx.load(onnx_file)
    match_pairs = []
    for node in model.graph.node:   
        if node.op_type == "Concat":
            qnodes = find_all_with_input_node(model, node.output[0])
            major = None
            for qnode in qnodes:
                if qnode.op_type != "QuantizeLinear":
                    continue
                conv = find_quantizelinear_conv(model, qnode)
                if major is None:
                    major = find_quantize_conv_name(model, conv.input[1])
                else:
                    match_pairs.append([major, find_quantize_conv_name(model, conv.input[1])])

                for subnode in model.graph.node:
                    if len(subnode.input) > 0 and subnode.op_type == "QuantizeLinear" and subnode.input[0] in node.input:
                        subconv = find_quantizelinear_conv(model, subnode)
                        match_pairs.append([major, find_quantize_conv_name(model, subconv.input[1])])


        elif node.op_type == "MaxPool":
            qnode = find_with_input_node(model, node.output[0])
            if not (qnode and qnode.op_type == "QuantizeLinear"):
                continue
            major = find_quantizelinear_conv(model, qnode)
            major = find_quantize_conv_name(model, major.input[1])
            same_input_nodes = find_all_with_input_node(model, node.input[0])

            for same_input_node in same_input_nodes:
                if same_input_node.op_type == "QuantizeLinear":
                    subconv = find_quantizelinear_conv(model, same_input_node)
                    match_pairs.append([major, find_quantize_conv_name(model, subconv.input[1])])
                   
    return match_pairs
