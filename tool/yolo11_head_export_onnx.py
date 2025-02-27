import argparse
from ultralytics import YOLO
import torch
import onnx
import onnxsim
import yaml
from attatch_nms_plugin import ModifiedDetectHead, attach_nms_node

def parse_args():
    parser = argparse.ArgumentParser(description='Export yolo11 model trained on coco.')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--attach_nms', action="store_true", default=False, help='Attach modified post_process nodes')
    parser.add_argument('--dynamic_batch_size', action="store_true", default=False, help='Use dynamic batch size')
    parser.add_argument('--dynamic_img_size', action="store_true", default=False, help='Use dynamic img size')
    parser.add_argument('--weights', dest='weights', type=str, default='../assets/yolo11s.pt', help='Path of trained model weights.')
    parser.add_argument('--config', default='../config/plugin_config.yml', help='Plugin config file used in end2end')
    parser.add_argument('--batch_size', type=int, default=1, help='Model input batch size, unused if dynamic_batch_size is set true')
    parser.add_argument('--input_h', type=int, default=640, help='Model input width, unused if dynamic_img_size is set true')
    parser.add_argument('--input_w', type=int, default=640, help='Model input height, unused if dynamic_img_size is set true')
    parser.add_argument('--output', default='../assets', help='Output onnx file path')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cpu" if args.cpu else "cuda")

    # create model and load weights
    model = YOLO(args.weights).model.to(device)
    model.model[-1].export=True
    model.model[-1].format="onnx"
    if args.attach_nms:
        setattr(model.model[-1], '__class__', ModifiedDetectHead)
    
    # export the model to ONNX format
    onnx_file_path_prefix = args.output+"/yolo11s_"+str(args.input_h)+"x"+str(args.input_w)
    if args.dynamic_batch_size:
        if args.dynamic_img_size:
            onnx_file_path_prefix = args.output+"/yolo11s_"+"nx3xhxw"
        else:
            onnx_file_path_prefix = args.output+"/yolo11s_"+"nx3x"+str(args.input_h)+"x"+str(args.input_w)
    else:
        if args.dynamic_img_size:
            onnx_file_path_prefix = args.output+"/yolo11s_"+str(args.batch_size)+"x3xhxw"
        else:
            onnx_file_path_prefix = args.output+"/yolo11s"+str(args.input_h)+"x"+str(args.input_w)
    onnx_file_path = onnx_file_path_prefix + "_raw.onnx"
    dummy_input = torch.randn((args.batch_size, 3, args.input_h, args.input_w)).to(device)
    if (args.dynamic_batch_size):
        torch.onnx.export(
            model,                   # model to be exported
            dummy_input,               # example input tensor
            onnx_file_path,            # file where the model will be saved
            export_params=True,        # store the trained parameter weights inside the model file
            opset_version=15,          # ONNX version to export the model to
            do_constant_folding=True,  # whether to perform constant folding for optimization
            input_names=['input'],     # name of the input tensor
            output_names=['output'],   # name of the output tensor
            dynamic_axes={'input': {0: 'batch_size', 2: 'h', 3: 'w'}}
        )
    else:
        torch.onnx.export(
            model,                   # model to be exported
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

    if args.attach_nms:
        with open(args.config, "r") as file:
            attr_config = yaml.load(file, Loader=yaml.SafeLoader)
        if args.dynamic_batch_size:
            onnx_final    = attach_nms_node(onnx_simp, 'batch_size', attr_config['nms_plugin'])
            onnx.save(onnx_final, onnx_file_path_prefix+".onnx")
        else:
            onnx_final    = attach_nms_node(onnx_simp, args.batch_size, attr_config['nms_plugin'])
            onnx.save(onnx_final, onnx_file_path_prefix+".onnx")
    else:
        onnx.save(onnx_simp, onnx_file_path_prefix+".onnx")
    print("plugin nms node attatch completed")
    print("model saved to "+onnx_file_path_prefix+".onnx")
