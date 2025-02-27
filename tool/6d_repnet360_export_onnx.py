import math
import os
import argparse

import numpy as np
from torchvision.utils import make_grid

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
import utils
from torch.hub import load_state_dict_from_url

import onnx
import onnxsim

class SixDRepNet360(nn.Module):
    def __init__(self, block, layers, fc_layers=1):
        self.inplanes = 64
        super(SixDRepNet360, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)

        self.linear_reg = nn.Linear(512*block.expansion,6)
      
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.linear_reg(x)        
        out = utils.compute_rotation_matrix_from_ortho6d(x)

        return out

def parse_args():
    parser = argparse.ArgumentParser(description='Export head pose estimation model using the Hopenet network')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--weights', dest='weights', type=str, default='./models/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth', help='Path of trained model weights.')
    parser.add_argument('--modified_post', action="store_true", default=False, help='Attach modified post_process nodes')
    parser.add_argument('--dynamic_batch_size', action="store_true", default=False, help='Use dynamic batch size')
    parser.add_argument('--batch_size', type=int, default=1, help='Model input batch size')
    parser.add_argument('--input_h', type=int, default=224, help='Model input width')
    parser.add_argument('--input_w', type=int, default=224, help='Model input height')
    parser.add_argument('--output', default='./models', help='Output onnx file path')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True

    device = torch.device("cpu" if args.cpu else "cuda")
    model = SixDRepNet360(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 6).to(device)
    model.eval()

    # Load snapshot
    if args.weights=='':
        saved_state_dict = load_state_dict_from_url("https://cloud.ovgu.de/s/TewGC9TDLGgKkmS/download/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth")    
    else:
        saved_state_dict = torch.load(args.weights)
    if 'model_state_dict' in saved_state_dict:
        model.load_state_dict(saved_state_dict['model_state_dict'])
    else:
        model.load_state_dict(saved_state_dict)  
    
    total = 0
    yaw_error = pitch_error = roll_error = .0
    v1_err = v2_err = v3_err = .0

    dummy_input = torch.randn((args.batch_size, 3, args.input_h, args.input_w)).to(device)
    if args.dynamic_batch_size:
        onnx_file_path_prefix = args.output+"/6DRepNet360"+"_"+"nx"+str(args.input_h)+"x"+str(args.input_w)
        onnx_file_path = onnx_file_path_prefix + "_raw.onnx"
        torch.onnx.export(
            model,                     # model to be exported
            dummy_input,               # example input tensor
            onnx_file_path,            # file where the model will be saved
            export_params=True,        # store the trained parameter weights inside the model file
            opset_version=15,          # ONNX version to export the model to
            do_constant_folding=True,  # whether to perform constant folding for optimization
            input_names=['input'],     # name of the input tensor
            output_names=['output'],   # name of the output tensor
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
    else:
        onnx_file_path_prefix = args.output+"/6DRepNet360"+"_"+str(args.batch_size)+"x"+str(args.input_h)+"x"+str(args.input_w)
        onnx_file_path = onnx_file_path_prefix + "_raw.onnx"
        torch.onnx.export(
            model,                     # model to be exported
            dummy_input,               # example input tensor
            onnx_file_path,            # file where the model will be saved
            export_params=True,        # store the trained parameter weights inside the model file
            opset_version=15,          # ONNX version to export the model to
            do_constant_folding=True,  # whether to perform constant folding for optimization
            input_names=['input'],     # name of the input tensor
            output_names=['output'],   # name of the output tensor
        )
    onnx_raw = onnx.load(onnx_file_path)
    onnx_simp, check = onnxsim.simplify(onnx_raw)
    onnx.save(onnx_simp, onnx_file_path_prefix+".onnx")