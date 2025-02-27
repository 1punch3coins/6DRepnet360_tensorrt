import torch
from ultralytics.nn.modules import Detect
from ultralytics.utils.tal import make_anchors
import numpy as np
import onnx_graphsurgeon as gs

class ModifiedDetectHead(Detect):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        dbox = dbox.permute((0,2,1))
        cls_score = cls.sigmoid().permute(0,2,1)
        return dbox, cls_score

@gs.Graph.register()
def attach_nms_plugin(self, inputs, outputs, attr_config):
    for inp in inputs:
        inp.outputs.clear()
    for out in outputs:
        out.inputs.clear()

    # check at https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin/efficientNMSPlugin.cpp#L413
    # and https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin/efficientNMSPlugin.cpp#L443
    op_attrs = dict()
    op_attrs["score_threshold"] = attr_config["score_threshold"]
    op_attrs["iou_threshold"] = attr_config["iou_threshold"]
    op_attrs["max_output_boxes"] = attr_config["max_output_boxes"]
    op_attrs["background_class"] = attr_config["background_class"]
    op_attrs["score_activation"] = attr_config["score_activation"]
    op_attrs["class_agnostic"] = attr_config["class_agnostic"]
    op_attrs["box_coding"] = attr_config["box_coding"]

    return self.layer(name="EfficientNMSPlugin_0", op="EfficientNMS_TRT", inputs=inputs, outputs=outputs, attrs=op_attrs)

def attach_nms_node(onnx_model, batch_size, attr_config):
    print("Use onnx_graphsurgeon to adjust postprocessing part in the onnx...")
    graph = gs.import_onnx(onnx_model)
    N = batch_size
    L = attr_config["max_output_boxes"]

    # check at https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin/README.md
    # and https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin/efficientNMSPlugin.cpp#L200
    out0 = gs.Variable(name="num", dtype=np.int32, shape=(N, 1))
    out1 = gs.Variable(name="boxes", dtype=np.float32, shape=(N, L, 4))
    out2 = gs.Variable(name="scores", dtype=np.float32, shape=(N, L))
    out3 = gs.Variable(name="classes", dtype=np.int32, shape=(N, L))

    graph.attach_nms_plugin(graph.outputs, [out0, out1, out2, out3], attr_config)
    graph.outputs = [out0, out1, out2, out3]
    graph.cleanup().toposort()
    return gs.export_onnx(graph)
