## Learning with Opponent-Learning Awareness

> Foerster, Jakob N., et al. "Learning with opponent-learning awareness." *arXiv preprint arXiv:1709.04326* (2017).



### 3. Notation

$$
G = \left\langle S,U,P,r,Z,O,n,\gamma \right\rangle ~~~~~~~~eq.1\\
a \in A \equiv \{1,...,n\} ,~\mathrm{n~agent} ~~~~~~~~eq.2\\
u^a \in U,~\mathrm{agent~action} ~~~~~~~~eq.3\\ 
s\in S ,~\mathrm{state~of~the~environment} ~~~~~~~~eq.4\\
u \in U, P\left(s'|s, u \right) ~~ \mathrm{joint~action~leads~state~trasition~ probability} ~~~~~~~~eq.5\\
P\left(s'|s, u \right) : S \times U \times S \rightarrow[0,1] ~~~~~~~~eq.6\\
r^a(s,u) : S\times U \rightarrow \mathbb{R},~ \mathrm{reward~function} ~~~~~~~~eq.7\\
\gamma \in [0,1) , ~ \mathrm{discount~factor} ~~~~~~~~eq.8\\
R_t^a = \sum_{l=0}^\infty \gamma^lr_{t+l}^a,~\mathrm{each~agent~}  ~~~~~~~~eq.9\\
$$

### 4. METHODS

4.1, 4.2에서 예상한 reward의 정확한 gradient와 hessians을 구할 수 있을 때 업데이트 규칙을 도출한다.

4.3에서는 policy gradients를 기반으로 학습 규칙을 만들기 때문에, gradient와 hessian을 제거할 수 있어짐.
하지만 agent가 상대방의 policy parameters에 접근할 수 있음

4.4에서는 LOLA 학습 규칙에서 상대방의 policy parameter를 경헙적으로 추론함으로 opponent modeling을 함

4.5에서는 고차 LOLA에 대해 discussion

### 4.1 Naive Learner

