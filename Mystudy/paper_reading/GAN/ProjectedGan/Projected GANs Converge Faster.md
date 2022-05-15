# Projected GANs Converge Faster

> Sauer, Axel, et al. "Projected gans converge faster." *Advances in Neural Information Processing Systems* 34 (2021).

## 1. Introduction

![image-20220514144311521](C:\Users\hyunsoo\AppData\Roaming\Typora\typora-user-images\image-20220514144311521.png)

GAN에서 discriminator의 경우 실제 이미지와 생성된 가짜 이미지를 data의 manifold 내에서 분류함으로 학습하고, 이를 통해 data의 representation을 학습한다.

하지만 GAN은 generator와 discriminator를 같이 학습 시키는 것은 어렵기 때문에, discriminator에 gradient penalties 같은 정규화 방법을 통해서 둘의 balance를 맞추지만  이는 하이퍼 파라미터에 영향을 받을 수 있고 성능저하를 만들어 낼 수도 있다.

이 논문에서는 gan의 학습 과정을 안정화시키기 위해서 pretrained representation을 사용하는 것에 대해서 설명한다.

**이전 computer vision이나 natural language processing에서 pretrained representation을 사용하는 것이 보편화 되어 있지만, pretrained networks를 gan에 사용하는 것은 noise-to-image synthesis에서 좋은 결과를 보이지 않았고(이미지의 다양성?), 이를 discriminator에 적용할 경우 discriminator가 dominate 하여 generator의 gradient vanishing이 발생함을 확인할 수 있다.**

위를 해결하기 위해서 multi-scale에서의 feature pyramids와 random projection을 통해 pretrained network를 잘 활용할 수 있음을 보여준다.

## 2. Related Work

**Pretrained Models for GAN Training**

gan에서 transfer learning task는 크게 2가지 카테고리가 있음

1.  기존 학습된 gan에 새로운 dataset을 적용하여 transfer
2.  pretrained model을 gan을 제어하거나 성능 향상에 사용함. (ex. styleclip, )
   이 방식은 pretrained model을 adversarial training이 필요가 없음.

**Discriminator Design**

Discriminator를 여러개 사용하는 것은 sample diversity, training stability에 도움이 되지만, computing resource 증가에 비해서 큰 도움이 되지 않아서 sota에선 많이 사용되지 않았음. 하지만 

MSGGan과 같이 multi scale feedback은 image sysnthesis나 image translation에서 도움이 된 것을 확인할 수 있다. 

## 3. Projected GANs

$$
\underset{G}{min} \underset{D}{max}
\left(
	\mathbb E_x \left[log D(x) \right] + 
	\mathbb E_z \left[log(1-D(G(z)))\right]
\right)
$$

일반 gan은 위와 같은 minmax objective를 통해서 training data의 분포를 학습한다고 볼 수 있다.
$$
\underset{G}{min} \underset{\{D_l\}}{max}
\sum_{l\in L}
\left(
	\mathbb E_x \left[log D_l(P_l(x)) \right] + 
	\mathbb E_z \left[log(1-D_l(P_l(G(z))))\right]
\right)
$$
P는 각 해상도별 feature projection이라고 볼 수 있는데, 이는 image를 project하는 역할을 하기 때문에 고정된 상태이며, 미분 가능하고, 입력에 대한 통계정보를 충분히 보존해야한다.

이를 통해 gan은 입력 data의 분포를 학습하는 것이 아닌 projected feature를 학습하게 될 수 있다.

##### 3.1 Consistency

