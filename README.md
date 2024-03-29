# PavementCrackDetection

### Crack and Sealed Crack Dataset 

We developed a highway asphalt pavement dataset containing 10,400 images captured by a highway condition monitoring vehicle with 202,840 labeled crack and sealed crack instances.

Please pay attention to the disk capacity when downloading.

[All images and labels](https://chdeducn-my.sharepoint.com/:u:/g/personal/2018024008_chd_edu_cn/EcW64xTdAfJEhK_mDgK5ArMBPF6UQLewa5oILlPXpVHDVg?e=ZdqWRn) contain all the 10400 images and their labels.

[Val](https://chdeducn-my.sharepoint.com/:u:/g/personal/2018024008_chd_edu_cn/EcZoz4-9QiZEjtVtD26pRl0Bv0NbGmhZ4Z7OskyU27TyHw?e=xA9fHQ) is just the validation set that  produced the results of our experiments.



### Trained Models

On the dataset mentioned above, we trained 13 currently prevalent object detection models from scratch, and the trained weights can be downloaded.

| Model(source)                                                | Trained Weights(on our dataset)                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [fasterrcnn_resnet50_fpn](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn) | [link](https://drive.google.com/file/d/1WU8hfkry_1e4LEm1R7qnpa9OWg-gX2PK/view?usp=sharing) |
| [fasterrcnn_resnet50_fpn_v2](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html#torchvision.models.detection.fasterrcnn_resnet50_fpn_v2) | [link](https://drive.google.com/file/d/1TvuMqAhZwknGYXpjQysqukz800IJKT2e/view?usp=sharing) |
| [fasterrcnn_mobilenet_v3_large_fpn](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn.html#torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn) | [link](https://drive.google.com/file/d/1vPEXH8G2msnU1o1iN5XamanSb09uM6kj/view?usp=sharing) |
| [fasterrcnn_mobilenet_v3_large_320_fpn](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html#torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn) | [link](https://drive.google.com/file/d/1o3_cs9774h109Mq3phgYCLeK3u-raevj/view?usp=sharing) |
| [fcos_resnet50_fpn](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fcos_resnet50_fpn.html#torchvision.models.detection.fcos_resnet50_fpn) | [link](https://drive.google.com/file/d/1PRwwF8cil-e7BtsGA3eIq47_PotuPYjQ/view?usp=sharing) |
| [retinanet_resnet50_fpn](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.retinanet_resnet50_fpn.html#torchvision.models.detection.retinanet_resnet50_fpn) | [link](https://drive.google.com/file/d/14ZPZ39yHq7egN3WlN8ug0Z1GVDB0cFfl/view?usp=sharing) |
| [retinanet_resnet50_fpn_v2](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.retinanet_resnet50_fpn_v2.html#torchvision.models.detection.retinanet_resnet50_fpn_v2) | [link](https://drive.google.com/file/d/16LHzaqeaiWZ7e-u7hxcfr2UumTAw-SmG/view?usp=sharing) |
| [ssd300_vgg16](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssd300_vgg16.html#torchvision.models.detection.ssd300_vgg16) | [link](https://drive.google.com/file/d/1W4w8dE65qKu--GEd6Ty5hmTCeJtMS3OJ/view?usp=sharing) |
| [ssdlite320_mobilenet_v3_large](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html#torchvision.models.detection.ssdlite320_mobilenet_v3_large) | [link](https://drive.google.com/file/d/1v_NppFISGRM0iAeNOZFzCpwSCXII0366/view?usp=sharing) |
| [yolov5n](https://github.com/ultralytics/yolov5)             | [link](https://drive.google.com/file/d/1pglkI2eMVzZdFNAt1_4Ep2CNbthPCPec/view?usp=sharing) |
| [yolov5s](https://github.com/ultralytics/yolov5)             | [link](https://drive.google.com/file/d/1qmFQdGkXUdoSSt2lGHLjrXpe8boKBooH/view?usp=sharing) |
| [yolov5m](https://github.com/ultralytics/yolov5)             | [link](https://drive.google.com/file/d/1M2YulCHkrGGMuzZK9mNxDD1bJKc3LxZE/view?usp=sharing) |
| [yolov5l](https://github.com/ultralytics/yolov5)             | [link](https://drive.google.com/file/d/1LnDkxlvCQqFuac-Z3tfT1ENjPg2OayXu/view?usp=sharing) |

All trained models are saved as checkpoints and could be loaded:

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

### Citation
```
@article{yang2022efficient,
  title={An Efficient Method for Detecting Asphalt Pavement Cracks and Sealed Cracks Based on a Deep Data-Driven Model},
  author={Yang, Nan and Li, Yongshang and Ma, Ronggui},
  journal={Applied Sciences},
  volume={12},
  number={19},
  pages={10089},
  year={2022},
  publisher={MDPI}
}
```
