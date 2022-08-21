# PavementCrackDetection

### Crack and Sealed Crack Dataset 

We developed a highway asphalt pavement dataset containing 10,400 images captured by a highway condition monitoring vehicle with 202,840 labeled crack and sealed crack instances.

Please pay attention to the disk capacity when downloading.

[All images and labels](https://drive.google.com/file/d/12hEIcr7sL1VHbyX0xdP_aFMX-U5nh0Hj/view?usp=sharing) contain all the 10400 images and their labels.

[Val](https://drive.google.com/file/d/1L1RfdCN_Os66l5S5EpJFIX2JA_1T4HD9/view?usp=sharing) is just the validation set that  produced the results of our experiments.



### Trained Models

On the dataset mentioned above, we trained 13 currently prevalent object detection models from scratch, and the trained weights can be downloaded.

| Model(source)                                                | Trained Weights(on our dataset) |
| ------------------------------------------------------------ | ------------------------------- |
| [fasterrcnn_resnet50_fpn](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn) |                                 |
| [fasterrcnn_resnet50_fpn_v2](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html#torchvision.models.detection.fasterrcnn_resnet50_fpn_v2) |                                 |
| [fasterrcnn_mobilenet_v3_large_fpn](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn.html#torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn) |                                 |
| [fasterrcnn_mobilenet_v3_large_320_fpn](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html#torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn) |                                 |
| [fcos_resnet50_fpn](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fcos_resnet50_fpn.html#torchvision.models.detection.fcos_resnet50_fpn) |                                 |
| [retinanet_resnet50_fpn](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.retinanet_resnet50_fpn.html#torchvision.models.detection.retinanet_resnet50_fpn) |                                 |
| [retinanet_resnet50_fpn_v2](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.retinanet_resnet50_fpn_v2.html#torchvision.models.detection.retinanet_resnet50_fpn_v2) |                                 |
| [ssd300_vgg16](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssd300_vgg16.html#torchvision.models.detection.ssd300_vgg16) |                                 |
| [ssdlite320_mobilenet_v3_large](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html#torchvision.models.detection.ssdlite320_mobilenet_v3_large) |                                 |
| [yolov5n](https://github.com/ultralytics/yolov5)             |                                 |
| [yolov5s](https://github.com/ultralytics/yolov5)             |                                 |
| [yolov5m](https://github.com/ultralytics/yolov5)             |                                 |
| [yolov5l](https://github.com/ultralytics/yolov5)             |                                 |

All trained models are saved with checkpoints and could be loaded as:

```python
import torch
import torchvision
# model
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(num_classes=3, box_score_thresh=0.25, box_nms_thresh=0.5)
# load checkpoint
checkpoint = torch.load("./path/to/checkpoint.pth", map_location="cpu")
# load trained weights
model.load_state_dict(checkpoint["model"])
```

