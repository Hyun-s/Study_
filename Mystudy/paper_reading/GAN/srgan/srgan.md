## Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network

> Ledig, Christian, et al. "Photo-realistic single image super-resolution using a generative adversarial network." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2017.

#### Introduction

supervised SR은 복구된 영상과 실제 영상 사이의 mse를 최소화 하는것이다.
이는 mse를 최소화 함으로 psnr도 최대화 되기 때문에 편리함.

하지만 복구된 영상의 texture는 픽셀 단위로 정의 되기 때문에 제한적이며, 가장 높은 psnr은 지각적으로 좋은 결과를 반드시 반영하지는 않는다.
$$
MSE = {1 \over XY}\sum_{x=0}^{X-1}\sum_{y=0}^{Y-1}e(x,y)^2\\
PSNR = 10~log{s^2\over MSE}
$$
srgan에서는 ResNet을 사용하고 mse가 아닌 gan을 최적화 대상으로 제안한다.
추가로 vgg네트워크의 high level feature maps 을 새로운 loss function으로 사용하게 된다.

#### Related work

prediction based method는 에지 보존 기반으로 초점이 맞춰져 있다.

**Design of convolutional neural networks**

deep nn 학습을 위해 batch normalization, residual block을 사용.



#### method

srgan 의 목표는 generating function G를 LR image로부터 그에 상응하는 HR Image를 추정하는 것이다.

**adversarial loss**

<img src="C:\Users\hyunsoo\AppData\Roaming\Typora\typora-user-images\image-20220122171438357.png" alt="image-20220122171438357" style="zoom:50%;" />

**Adversarial network architecture**

B개의 residual block은 3,3 kernel, 64 feature map으로 이루어진 2개의 cnn layer, BN, PReLU로 이루어져 있다.

영상 해상도의 크기 증가를 위해서 2개의 학습된 sub-pixel cnn layer를 사용한다.

disc 는 vgg19로 생성된 512 featuremap으로부터 2개의 dense layer, final sigmoid를 사용하여 분류함

**Perceptual loss function**
$$
l^{SR} = l_X^{SR} + 10^{-3}l_{gen}^{SR}
$$
좌항은 content loss(pixel로 비교했을 때 원래 영상과 얼마나 유사한지),
우항은 adversarial loss(gan의 loss로 영상 생성의 품질?)를 의미

**Content loss**
$$
l_{MSE}^{SR} = {1 \over r^2WH}\sum_{x=1}^{rW}\sum_{y=1}^{rH}\left(I_{x,y}^{HR} - G_{\theta G}\left(I^{LR} \right)_{x,y} \right)^2
$$
pixel 기반의 mse loss
이는 PSNR을 향상시키는데 좋은 효과를 보이지만, 고주파 성분에 대한 정보가 blur하게 됨을 보임

개선을 위해 인지적 유사도와 가장 근사한 loss function은 아래의 ReLU 기반의  VGG loss를 새로 정의함
$$
l_{VGG/i.j}^{SR} = 
{1 \over W_{i,j}H_{i,j}}
\sum_{x=1}^{W_{i,j}}\sum_{y=1}^{H_{i,j}}
\left(
	\phi_{i,j}
	\left( I^{HR}
	\right)_{x,y}
	-
	\phi_{i,j}
	\left(
		G_{\theta_G}
		\left(I^{LR}\right)
	\right)_{x,y}
\right)^2
$$
W_i,j와 H_i,j 는 vgg의 feature map과 관련딘 dimension을 의미한다. (vgg19 일 때 i=1, j=9)

**Adversarial loss**

자연 영상의 manifold의 학습을 위해 adversarial loss 를 추가한다.
$$
l_{gen}^{SR} = \sum_{n=1}^N-logD_{\theta_D}
\left(
	G_{\theta_G}
	\left(
		I^{LR}
	\right)
\right)
$$


