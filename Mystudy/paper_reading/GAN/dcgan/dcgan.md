## UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS

#### ABSTRACT

cnn을 활용하여 deep convolutional gan 구축

#### 1. INTRODUCTION

비지도 학습에서 unlabeled dataset의 재사용 가능한 특징 추출이 필요함.

즉, unlabeled image를 활용하여 특징 표현을 한다면, 이미지 분류 등의 지도학습에서 사용될 수 있다.

본 논문에서는 이를 위해 gan을 활용하여 image representation을 학습 하는 방법을 제안하였다.

- 이미지 분류에 학습된 분류기를 사용하고, 비지도 학습과 비교하여 경쟁력 있는 성능이 나올지 보여줌.
- gan이 학습된 필터를 시각화하고, 특정 필터가 특정 객체를 그리도록 학습함을 보여줄 수 있음.
- generator가 생성된 샘플의 특성들을 쉽게 조작할 수 있고, 쉽게 조작 가능함.

#### 2. RELATED WORK

###### 2.1 REPRESENTATION LEARNING FROM UNLABELED DATA

unsupervised representation learning에 대한 이전 방법들은 k-means, image patch의 계층적 군집화, auto encoder 등이 있다.

###### 2.2 GENERATING NATURAL IMAGES



###### 2.3 VISUALIZING THE INTERNALS OF CNNS

#### 3. APPROACH AND MODEL ARCHITECTURE

본 논문에서의 3가지 중요한 접근방식은 아래와 같다.

1. maxpooling 등의 기능을 stride convolution을 사용하여 down sampling을 학습한다.
2. convolutional feature 위의 fully connected layer를 제거한다.
3. batch normalization을 사용한다. batch normalization은 각 유닛의 입력을 정규화 하여 각 층의 잘못된 초기화로 발생하는 문제와, 더 많은 층을 가진 모델에서 기울기를 전파하는데 도움을 주어 학습을 안정화시킬 수 있다. 하지만, gan에서 모든 층에 적용한다면, sample oscillation과 model instability의 문제가 발생하여, generator의 output과 discriminator의 input엔 사용하지 않는다.

generator에서는 relu를 사용하고 마지막 layer만 tanh를 사용하였다. 이는 generator가 색상 공간에 있어 빨리 학습하는데 도움이 되었다.
discriminiator에서는 leaky relu를 사용하였고, 마지막 layer에서는  sigmoid로 사용하였다. 

#### 4. DETAILS OF ADVERSARIAL TRAINING

dcgan을 학습하는데 있어 논문에서는 image를 generator의 output의 범위인 [-1,1] 로 scaling을 진행하였으며, 128의 mini batch, sgd(학습률 0.0002)를 사용했다고 한다. 또한 momentum term을  0.9가 아닌 0.5를 사용하는 것이 학습의 안정성에 있어서 도움이 되었다고 한다.

###### 4.1 LSUN

학습은 overfitting/memorizing training examples 가 발생하면 안됨.

###### 4.1.1 DEDUPLICATION

###### 4.2 FACES

###### 4.3 IMAGENET-1K

#### EMPIRICAL VALIDATION OF DCGANs CAPABILITIES

###### 5.1 CLASSIFYING CIFAR-10 USING GANs AS A FEAURES EXTRACTOR

