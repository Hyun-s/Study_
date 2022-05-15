## Generative Modeling by Estimating Gradients of the Data Distribution

> Song, Yang, and Stefano Ermon. "Generative modeling by estimating gradients of the data distribution." *Advances in Neural Information Processing Systems* 32 (2019).

#### 1. Introduction

최근 generative model의 경우 크게 likelihood-based method, gan 이 두가지로 많은 연구들이 이루어져 있음.

VAE와 같은 likelihood-based model과 같은 경우 ELBO와 같은 대체 loss를 사용하거나, gan의 경우 adversarial training으로 인해 학습이 불안정하거나, 다른 gan과의 비교가 쉽지 않다는 문제점들이 있다.

