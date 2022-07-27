## QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning

> Rashid, Tabish, et al. "Qmix: Monotonic value function factorisation for deep multi-agent reinforcement learning." *International Conference on Machine Learning*. PMLR, 2018.

#### Abstract

실제 상황은 decentralized 방법으로 행동하지만, 학습 시에는 centralized하게 학습할 수 있는 경우에 대해 적용

QMIX에선 centralized하게 학습시켰을 때, decentralized policy를 추출하는 방법에 대하여 제시함.

#### 1. Introduction

#### 4. QMIX

QMIX는 각 agent의 Q value를 mixing network를 통해 전체 Q value를 계산하는 방법이다.

<img src="C:\Users\hyunsoo\AppData\Roaming\Typora\typora-user-images\image-20220606204215432.png" alt="image-20220606204215432" style="zoom:67%;" />

위는 marl에서의 joint-action value Function으로 global 하게 argmax(Q)한 결과와 개별로 실행한 결과가 같아야 한다는 의미가 된다. (각 agent의 최선의 선택이 모두에게 최선의 선택이 된다.) VDN에서는 이를  각 utility의 합으로 구성하지만, QMIX에서는 위의 Q_tot를 mixing network를 통하여 계산하기 때문에, 위의 joint-action value Function을 만족시키기 어렵다. 그렇기 때문에 아래와 같이 Q_a와 Q_tot에 대해서 monotonic하게 만드는 제약조건을 적용하였다.

$$
{\partial Q_{tot} \over \partial Q_a} \geq 0, \forall a \in A
$$

**QMIX architecture**

![image-20220606203211817](C:\Users\hyunsoo\AppData\Roaming\Typora\typora-user-images\image-20220606203211817.png)

QMIX를 도식화 한 그림이다. 여기서 QMIX의 mixing network를 표현한 그림은 (a) 로 빨간 박스가 hyper network를 의미하며, 파란 박스는 mixing network를 의미한다.

각 hyper network는 state를 입력으로 받아 mixing network의 가중치를 생성하게 되는데, hyper network는 single linear layer로 absolute activation function을 가진다. 즉 hyper network는 각 state를 linear layer를 통하여 mix하는 효과를 가진다. linear layer로 생성되기 때문에, output의 vector형태로 나오게 되는데, 이를 matrix로 reshape를 하게 된다. 

이를 통해 mixing network의 가중치를 생성한 후 위의 Q_a와 Q_tot는 monotonic하다는 조건을 만족하기 위한 방법으로 간단히 mixing network의 가중치들을 절대값으로 설정하였다.

결론적으로 QMIX의 end-to-end loss는 아래와 같다.
$$
\mathcal{L}(\theta) = \sum_{i=1}^b \left[
										\left(
											y_{i}^{tot} - Q_{tot}({\tau,\mathbf u,s;\theta})
										\right)^2
								   \right]\\
y^{tot} = r + \Upsilon max_{u'}Q_{tot}({\tau',\mathbf u',s';\theta^-})
$$



#### implement

mixing network를 projected gan에서의 cross channel mixing 사용해보기

