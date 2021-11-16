# LSTM: A Search Space Odyssey

Greff, Klaus, et al. "LSTM: A search space odyssey." *IEEE transactions on neural networks and learning systems* 28.10 (2016): 2222-2232.

## Vanilla LSTM

<img src="C:\Users\hyunsoo\AppData\Roaming\Typora\typora-user-images\image-20211114190538612.png" alt="image-20211114190538612" style="zoom:70%;" />
$$
input : W_z,\ W_i,\ W_f,\ W_o \in \mathbb{R}^{N \times M} \\
Recurrent : R_z,\ R_i,\ R_f,\ R_o \in \mathbb{R}^{N \times M}\\
Peephole : p_i,\ p_f,\ p_o\in \mathbb{R}^{N}\\
Bias : b_z,\ b_i,\ b_f,\ b_o \in \mathbb{R}^{N}
$$
