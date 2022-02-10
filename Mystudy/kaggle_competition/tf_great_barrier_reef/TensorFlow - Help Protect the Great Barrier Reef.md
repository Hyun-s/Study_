## TensorFlow - Help Protect the Great Barrier Reef

#### Competition

**Goal**

Great barrier reef는 다른 해양 생물들의 서식지로 가시나무 불가사리(cots)에 위협받고 있고, 이를 실시간으로 정확히 파악하는 것이 목표이다

**Evaluation**

이 competition은 여러  intersection over union(IoU) thresholds에 따라 F2-score를 평가지표로 사용한다.
$$
f2~score = {(5\times Precision \times Recall) \over (4 \times Precision + Recall)}
$$
<img src="https://blog.kakaocdn.net/dn/wNXOK/btqSpGVHmHc/KbsxRBSs6KymYB3PkEny21/img.png" alt="img" style="zoom:80%;" />

이 때 IoU 의 threshold는 0.3~0.8, stepsize=0.05로 이루어진다.



## 🐡GreatBarrierReef: YOLO Full Guide [train+infer]

#### YOLO

**paremeters**

```python
FOLD      = 1 # which fold to train
DIM       = 1000 
MODEL     = 'yolov5s6'
BATCH     = 4
EPOCHS    = 7a
OPTMIZER  = 'Adam'

PROJECT   = 'great-barrier-reef-public' # w&b in yolov5
NAME      = f'{MODEL}-dim{DIM}-fold{FOLD}' # w&b for yolov5

REMOVE_NOBBOX = True # remove images with no bbox
ROOT_DIR  = '/content/data/'
IMAGE_DIR = '/content/images' # directory to save images
LABEL_DIR = '/content/labels' # directory to save labels
```





**Data preproecessing**

[yolo]( https://github.com/ultralytics/yolov5)를 사용하기 위해  기존의 data를 image,label이라는 추가적인 폴더 생성 후 image,label을 복사해야함

1. data는 bounding box가 coco format(기준 점은 우측 상단)으로 되어있으므로 yolo format(기준 점은 무게중심)으로 변경해주어야 함
   <img src="https://i0.wp.com/prabhjotkaurgosal.com/wp-content/uploads/2021/03/image.png?resize=1024%2C279&ssl=1" alt="img" style="zoom:80%;" />

2. images 폴더에 모든 객체가 존재하는 image 복사, labels 폴더에 label(bounding box)를 txt format으로 복사

3. working 폴더에 train, validation data의 path를 txt format으로 저장

   ![image-20220209170750666](C:\Users\hyunsoo\AppData\Roaming\Typora\typora-user-images\image-20220209170750666.png)![image-20220209170811341](C:\Users\hyunsoo\AppData\Roaming\Typora\typora-user-images\image-20220209170811341.png)

4. 현재 경로, train.txt, val.txt 등의 경로들 지정하는 yaml 파일 저장
   <img src="C:\Users\hyunsoo\AppData\Roaming\Typora\typora-user-images\image-20220209171147945.png" alt="image-20220209171147945" style="zoom:70%;" />

5. 학습시 지정할 hyperparameter을 저장할 hyp.yaml 저장

**hyperparameter 종류**

[참고](https://docs.ultralytics.com/tutorials/hyperparameter-evolution/)

```python
# Hyperparameter Evolution Results
# Generations: 1000
#                   P         R     mAP.5 mAP.5:.95       box       obj       cls
# Metrics:     0.4761      0.79     0.763    0.4951   0.01926   0.03286  0.003559

lr0: 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
lrf: 0.2  # final OneCycleLR learning rate (lr0 * lrf)
momentum: 0.937  # SGD momentum/Adam beta1
weight_decay: 0.0005  # optimizer weight decay 5e-4
warmup_epochs: 3.0  # warmup epochs (fractions ok)
warmup_momentum: 0.8  # warmup initial momentum
warmup_bias_lr: 0.1  # warmup initial bias lr
box: 0.05  # box loss gain
cls: 0.5  # cls loss gain, class에 대한 loss
cls_pw: 1.0  # cls BCELoss positive_weight
obj: 1.0  # obj loss gain (scale with pixels), 객체 탐지에 대한 loss
obj_pw: 1.0  # obj BCELoss positive_weight
iou_t: 0.20  # IoU training threshold
anchor_t: 4.0  # anchor-multiple threshold
anchors: 0  # anchors per output grid (0 to ignore)
fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5)
hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
hsv_v: 0.4  # image HSV-Value augmentation (fraction)
degrees: 0.0  # image rotation (+/- deg)
translate: 0.1  # image translation (+/- fraction)
scale: 0.5  # image scale (+/- gain)
shear: 0.0  # image shear (+/- deg)
perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
flipud: 0.0  # image flip up-down (probability)
fliplr: 0.5  # image flip left-right (probability)
mosaic: 1.0  # image mosaic (probability)
mixup: 0.0  # image mixup (probability)
```

**OneCycleLR**
	<img src="https://www.deepspeed.ai/assets/images/1cycle_lr.png" alt="1cycle_lr" style="zoom:67%;" />

**Focal Loss**

이는 data의 class imbalance 문제에 대한 개선 방안으로 기존의 cross entropy에 (1-p)^r을 곱함으로 easy example에 대한 loss는 크게 줄임으로 학습이 잘 된 case에 대해서 loss의 비중을 낮출 수 있다.

<img src="https://gaussian37.github.io/assets/img/dl/concept/focal_loss/1.png" alt="Drawing" style="zoom:80%;" />

**augmentation**

![Crop, Rotation, Flip, Hue, Saturation, Exposure, Aspect Ratio, MixUp, CutMix, Mosaic, Blur](https://blog.roboflow.com/content/images/2020/05/image-10.png)

