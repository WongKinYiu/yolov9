# yolov9
Implementation of YOLOv9

## PGI Type

| Model | Type | min/epoch | AP<sup>val</sup> | Param. | FLOPs |
| :-- | :-: | :-: | :-: | :-: | :-: |
| [**YOLOv9-C**]() | -- | 7 | **52.5%** | **25.3M** | **102.1G** |
| [**YOLOv9-C**]() | PFH | 8 | **52.5%** | **25.3M** | **102.1G** |
| [**YOLOv9-C**]() | FPN | 9 | **52.6%** | **25.3M** | **102.1G** |
| [**YOLOv9-C**]() | ICN | 11 | **52.9%** | **25.3M** | **102.1G** |
| [**YOLOv9-C**]() | LHG-ICN | 10 | **53.0%** | **25.3M** | **102.1G** |
| [**YOLOv9-C**]() | C2F-LHG-ICN (coarse branch) | -- | **52.7%** | **25.3M** | **102.1G** |
| [**YOLOv9-C**]() | C2F-LHG-ICN (fine branch) | -- | **52.4%** | **25.3M** | **102.1G** |

* LHG indicates lead head guide assignment proposed by YOLOv7, which use lead branch to make consistant label assignment for auxiliary branch.
* C2F indicates coarse to fine assignment proposed by YOLOv7, which enable fine branch could be NMS-free.
* Training time are estimated on RTX 6000 ada with batch size 128.

## Versatility of PGI on various architectures

| Model | AP<sup>val</sup> | w/o PGI | Param. | FLOPs |
| :-- | :-: | :-: | :-: | :-: |
| [**Anchor-based YOLOv9-C**]() | **50.7%** | 50.6% | **20.2M** | **78.4G** |
|  |  |  |  |  |
| [**Anchor-free YOLOv9-C**]() | **53.0%** | 52.5% | **25.3M** | **102.1G** |
| [**Mask-guided YOLOv9-C**]() | **53.3%** | 52.3% | **25.3M** | **102.1G** |
| [**Light Head YOLOv9-C**]() | **52.9%** | 52.3% | **21.1M** | **82.5G** |
| [**YOLOv9-C Lite**]() | **52.7%** (temp) | -- | **13.3M** | **66.7G** |
|  |  |  |  |
| [**Anchor-free YOLOv9-C (fine branch)**]() | **52.4%** | 52.3% | **25.3M** | **102.1G** |
| [**Mask-guided YOLOv9-C (fine branch)**]() | **52.7%** (temp) | -- | **25.3M** | **102.1G** |
| [**Light Head YOLOv9-C (fine branch)**]() | **52.7%** | 52.1% | **21.1M** | **82.5G** |
| [**YOLOv9-C Lite (fine branch)**]() | **52.4%** (temp) | -- | **13.3M** | **66.7G** |
|  |  |  |  |
<!-- | [**YOLOv9-C TR**]() |  **%** (temp) | **M** | **G** | -->
<!-- | [**YOLOv9-C TR (fine branch)**]() |  **%** (temp) | **M** | **G** | -->

* Anchor-based YOLOv9 replace YOLOv9 head by YOLOv7 head.
* Anchor-free YOLOv9 is YOLOv9.
* Mask-guided YOLOv9 use instance segmentation task to guide YOLOv9 training.
* Light Head YOLOv9 replace 3x3 convolution in YOLOv9 head by 3x3 depth-wise separable convolution.
* YOLOv9 Lite is depth-wise convolution-based YOLOv9.
* (temp) indicates not yet finish training.
<!-- * YOLOv9 TR is Transformer-based YOLOv9. -->

## Versatility of PGI on small dataset

| Model | PGI | Pretrained | AP<sup>val</sup> | Param. | FLOPs |
| :-- | :-: | :-: | :-: | :-: | :-: |
| [**YOLOv9-S**]() | -- | -- | **64.4%** | **7.1M** | **26.3G** |
| [**YOLOv9-S**]() | :heavy_check_mark: | -- | **65.1%** | **7.1M** | **26.3G** |
|  |  |  |  |  |
| [**YOLOv9-S**]() | -- | COCO | **73.5%** | **7.1M** | **26.3G** |
| [**YOLOv9-S**]() | :heavy_check_mark: | COCO | **74.4%** | **7.1M** | **26.3G** |

* We choose VOC as the small dataset.


