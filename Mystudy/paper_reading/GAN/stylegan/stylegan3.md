## Alias-Free Generative Adversarial Networks

> Karras, Tero, et al. "Alias-free generative adversarial networks." *Advances in Neural Information Processing Systems* 34 (2021).

#### Abstract

이전 계층적인 convolution을 통한 이미지 합성에서는 객체의 스타일이 객체의 표면에 있는 것이 아닌 이미지 좌표에 의존한다는 문제점이 있음

[참고 영상, 3:04~](https://www.youtube.com/watch?v=j1ZY7LInN9g)

#### 1. Introduction

**Texture sticking**

실제 영상의 경우는 객체의 움직임에 따라 객체 내부의 요소들이 같이 움직이거나, 크기가 변한다.

하지만, gan의 경우 객체가 움직임에도 특정 요소들이 픽셀에 고정되어 있는 현상이 발생한다.

![image-20220404173425700](C:\Users\hyunsoo\AppData\Roaming\Typora\typora-user-images\image-20220404173425700.png)

이에 대한 예시로, 이미지를 약간씩 움직이면서 생성한 이미지들의 평균을 본 결과, 이전 stylegan2에선 눈이 blur 되고 주변의 털과 같은 texture들은 선명하게 있음을 확인 가능하다. 이는 눈 주변의 털과 같은 경우 움직이지 않음을 의미한다.

이유는 generator의 convolution이 의도치 않게 positional reference를 참조하기 때문

**Unintentional positional reference**

1. image borders
   Convolution 연산 시 모서리 padding을 함으로 network가 모서리의 위치를 파악함.
   이러한 점은 간단히 이미지를 크게 만들고 crop하는 방식으로 해결 가능
   
2. per-pixel noise inputs

3. positional encoding > 위치 자체에 대한 인코딩을 사용

4. aliasing > 위치 변환 후 upsampling 할 때, 특정 부분만 aliasing 현상이 발생하여 sync가 맞지 않음.
   
   ![image-20220408121948284](C:\Users\hyunsoo\AppData\Roaming\Typora\typora-user-images\image-20220408121948284.png)
   upsampling filter의 interpolation 방식, ReLU나 swish의 픽셀별 비선형성
   => 여러 scale에서 texture의 패턴을 그리고 결합하면서 약간의 aliasing 현상 발생
   이는 bandlimited function을 활용하여 이산적인 sample grid에 대한 표현보다 연속적인 도메인으로 변환

**aliasing**

![image-20220408163545666](C:\Users\hyunsoo\AppData\Roaming\Typora\typora-user-images\image-20220408163545666.png)

aliasing은 특정 frequency를 sampling 할 때 적당하지 않은 sampling rate로 sampling하면 원래의 frequency를 복원 못하는 현상

[Making Convolutional Networks Shift-Invariant Again](https://www.youtube.com/watch?v=eZa56DqXTHg)

> Zhang, Richard. "Making convolutional networks shift-invariant again." *International conference on machine learning*. PMLR, 2019.

이처럼 이미지가 이동함에도 같은 결과를 내야하는 특징(equivariance)을 가져야하지만 실제로 다른 결과를 내보인다.

이 문제는 convolution이 아닌 pooling에 이유가 있음

![image-20220408163647600](C:\Users\hyunsoo\AppData\Roaming\Typora\typora-user-images\image-20220408163647600.png)

이러한 shift equivariance에 대해선 stride를 1로 maxpooling-> blur kernel -> sub sampling을 통해 개선한 선행 연구가 있음.



#### 2. Equivariance via continuous signal interpretation

![image-20220408173738148](C:\Users\hyunsoo\AppData\Roaming\Typora\typora-user-images\image-20220408173738148.png)

![image-20220408174831880](C:\Users\hyunsoo\AppData\Roaming\Typora\typora-user-images\image-20220408174831880.png)

이처럼 sampling과 reconstruction을 하는 filter를 잘 설계하면 equivariance를 만족 할 수 있다.

**2.1 Equivariant network layers**
$$
f \circ t = t \circ f
$$
위 식에서 f는 generator, t는 translation일 때, 연산 순서가 서로 바뀌어도 같은 결과가 나오는지 확인해야한다.

**Convolution**

![image-20220408180009243](C:\Users\hyunsoo\AppData\Roaming\Typora\typora-user-images\image-20220408180009243.png)

![image-20220408175937265](C:\Users\hyunsoo\AppData\Roaming\Typora\typora-user-images\image-20220408175937265.png)

직관적으로 convolution 연산은 kernel이 sliding 하면서 연산을 하기 때문에, translation에 equivariance특성을 가진다고 생각할 수 있다.

![image-20220408180154751](C:\Users\hyunsoo\AppData\Roaming\Typora\typora-user-images\image-20220408180154751.png)

rotation?

**Upsampling and Downsampling**

이상적인 upsampling은 Shannon-Nyquist에 맞는 sampling rate가 된다면 연속 공간 내에서 표현을 바꾸지 않지만,

**Nonlinearity**

![image-20220409161021009](C:\Users\hyunsoo\AppData\Roaming\Typora\typora-user-images\image-20220409161021009.png)

위처럼 ReLU activation의 경우 0 기준으로 nonlinear하기 때문에, 위 그림에서 가운데 실제 출력 부분과 같이 엣지가 생기게 된다. 이러한 엣지는 continuous domain에서 결과에서 표현될 수 없는 높은 frequency를 가지게 된다는 것이다. 하지만 discrete domain Z에서는 바로 직접적으로 연산이 불가함으로 z로 upsampling 한 후에 z' 이후 다시 downsampling 함으로 근사시킬 수 있다.

이는 low-pass filter를 적용함으로 z'과 같이 high frequency를 제거 가능하다.



#### 3. Practical application to generator network

위와 같은 방법을 적용한다면, generator는 w,rotation,translation에 대해 equivariant 특성을 가질 수 있을 것이다. 
$$
z_0: \mathbf g(\mathbf t[z_0];w) = \mathbf t[\mathbf g(z_0;w)]
$$
이에 대해서 평가하기 위해서 PSNR을 사용하여  평가할 수 있다.
$$
PSNR = 10\cdot log_{10} \left(MAX_i^2 \over MSE \right)\\
EQ-T = 10\cdot log_{10} 
\left(
	I_{max}^2 / \mathbb E_{\mathbf w\sim \mathcal W, x \sim \mathcal X^2,p \sim\mathcal V, c \sim \mathcal C}
	\left[
		 \left(
		 \mathbf
		 g(\mathbf t_x[z_0];\mathbf w)_c(p) - \mathbf t_x[\mathbf g(z_0;\mathbf w)]_c(p)
		 \right)^2
	\right]
\right)
$$


![image-20220409165456051](C:\Users\hyunsoo\AppData\Roaming\Typora\typora-user-images\image-20220409165456051.png)

![image-20220409184928675](C:\Users\hyunsoo\AppData\Roaming\Typora\typora-user-images\image-20220409184928675.png)

**Fourier features (B)**

이전 stylegan2 는 입력 벡터를 4x4x512로 사용했지만, 입력 벡터를 fourier features로 바꾸게 되었다.

이는 frequency domain이기 때문에 크기가 정해져 있는게 아닌 무한한 크기의 맵을 정의해줄 수 있다고 한다.

**No noise inputs (C)**

이는 이전의 per-pixel noise input은 translation과 상관이 없기 때문에 제거하였다고 한다.

**Simplified generator (D)**

이전 stylegan2에서 mapping network depth, mixing regularization, path length regularization을 줄이거나 사용하지 않았다.
그 이유로는 위의 방법들은 gradient에 대해서 FID를 높이기 위해서 사용했던 방법으로, 간단히 convolution 전에 nomalization을 함으로 간단히 해결하였다고 한다. 그렇기 때문에 상대적으로 FID가 높아짐을 확인할 수 있다.

**Boundaries and upsampling (E)** 

image borders에서 모서리를 통해 위치 정보를 알 수 있었으니, 더 큰 크기의 feature map을 연산 후 crop함.

이전의 bilinear upsampling 대신 kaiser window 기반의 filter를 만들어 convolution연산 함으로 upsamping한다.

**Filtered nonlinearities (F) **

ReLU와 같은 non-linear activation으로 인한 high frequency 문제에 대해서
m x upsampling -> ReLU -> m x downsampling 과 같은 방식으로 사용한다.

하지만 upsampling 하는 layer의 경우,
2m x upsampling -> ReLU -> m x downsampling  을 사용하여 upsampling을 진행한다.

이와 같은 block의 경우 custom cuda kernel를 따로 구현하여 학습 속도를 10배 빠르게 했다고 한다.

**Non-critical sampling (G)**

low-pass filter의 경우 Frequency response 그림을 보다시피 cut off frequency를 낮춘다.
$$
f_c = s / 2 -f_h
$$
이는 s/2보다 큰 frequency를 제거해버리게 되는데 이렇게 한다면 aliasing을 확실히 줄일 수 있다. 하지만 generator의 resolution이 높은 layer의 경우 명확한 이미지, 즉 high frequency에 대해서 생성해야 함으로 마지가장 큰 resolution을 가지는 layer를 제외하고 cut off frequency를 낮추었다고 한다.

**Transformed Fourier features (H)**

이는 추가적인 affine layer를 사용해서 w를 기준으로 (r_cos, r_sin, t_x, t_y)를 생성하도록 하여 rotation, translation과 같은 기하적 변환을 만들어 내는 것이다. 

이를 통해서 style에 해당했던 w와 fourier features간의 domain을 맞추는 것과 비슷한 역할을 한 것

**Flexible layer specifications (T)**

config G 와 같은 방법으로 equivariance quality에 대한 향상이 있었지만, 아직도 artifacts들이 존재했다고 한다.

그래서 낮은 resolution일 수록 aliasing을 없에기 위해 낮은 cut off를 적용해야 했으며, 
반대로, 높은 resolution일 수록 high frequency로부터 나오는 detail을 위해 높은 cut off를 적용해야 한다.

그래서 위의 그림 C와 같이 각 layer에서 f_c,f_h,f_t를 자동적으로 결정할 수 있게 한다.

**Rotation equivariance (R)**

rotation equivariance를 위해서 모든 convolution kernel의 크기를 3x3 에서 1x1로 감소시키고 feature map 수를 2배로 증가시켰다고 한다.

그 이유로는 3x3  kernel의 경우는 kernel 내에서 방향이 나오기 때문에 방향성이 없는 1x1을 사용한 것이다. 이로 인한 capacity 감소가 있기 때문에 feature map을 2배 증가시킨다. 
이는 fid나 eq-t 를 크게 해치지 않으면서 각 layer당 56%정도의 파라미터 감소를 만들 수 있다고 한다.

추가로 downsampling filter를 radially symmetric하게 만들어 어느 방향으로든 대칭이 될 수 있게 만들었다고 한다.

#### 4. Results

![image-20220409200601507](C:\Users\hyunsoo\AppData\Roaming\Typora\typora-user-images\image-20220409200601507.png)



#### 5. Limitations, discussion, and future work

generator 뿐만 아니라 discriminator도 equivariant하게 만들면 더욱 좋을 것이다.

noise input과 path length regularization을 equivariant를 고려하여 설계

