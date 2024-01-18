# SSP
An Object Detection Model for Remote Sensing Images Based on Attention Mechanism and Improved Downsampling Strategy.
# Abstract
Object detection is an essential task in remote sensing image processing. However, the remote sensing images are characterized by large range of object sizes and complex object backgrounds, which results in challenges in the object detection task. Moreover, the detection effect of existing object detectors on remote sensing images is still not satisfactory. In order to tackle the above problems, an object detection model named YOLO-SSP for remote sensing images is proposed based on the YOLOv8m model in this paper. To begin with, the original downsampling layers are replaced with the proposed lightweight SPD-Conv module, which performs downsampling without loss of fine-grained information and improves the ability of the network to learn the feature representation. In addition, to adapt the large number of small objects in remote sensing images, a small object detection layer is added and achieves the expected results. Finally, a pyramid spatial attention mechanism (PYSAM) is proposed to obtain the weights of different spatial positions through hierarchical pooling operations. It effectively improves the detection performance of small objects and those with complex backgrounds. We conducted ablation experiments on the DIOR dataset and compared the YOLO-SSP model with other state-of-the-art models. To demonstrate the generalizability and robustness of the improved model, the comparison experiments are also performed on the TGRS-HRRSD dataset and SIMD dataset.
# Datasets
The three datasets used in this research can be downloaded from [DIOR](http://www.escience.cn/people/gongcheng/DIOR.html),[TGRS-HRRSD](https://github.com/CrazyStoneonRoad/TGRS-HRRSD-Dataset),[SIMD](https://github.com/ihians/simd).

