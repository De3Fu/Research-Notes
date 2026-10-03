# 1. Denoising Diffusion Probabilistic Model

**Denoising Diffusion Probabilistic Model（DDPM，去噪扩散概率模型）**主要包含两个过程：

- Forward Process：前向加噪
- Reverse Process：反向去噪 / 生成

---

## 1.1 Forward Diffusion Process

DDPM 的前向过程会不断向原始图像中添加高斯噪声。

首先定义：

$$\alpha_t=1-\beta_t$$

其中 $\beta_t$ 表示第 $t$ 个时间步加入的噪声强度。

单步前向扩散过程可以写成：

$$x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon$$

其中：

$$\epsilon\sim\mathcal{N}(0,I)$$

也可以写成概率分布：

$$q(x_t\mid x_{t-1})=\mathcal{N}\left(\sqrt{\alpha_t}x_{t-1},(1-\alpha_t)I\right)$$

---

## 1.2 从 $x_0$ 直接计算 $x_t$

定义累计乘积：

$$\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s$$

由于 DDPM 的前向扩散过程属于**线性高斯 Markov 过程**，连续多个高斯加噪步骤可以合并。

因此，我们不需要真的执行：

$$x_0\rightarrow x_1\rightarrow x_2\rightarrow\cdots\rightarrow x_t$$

而可以直接从 $x_0$ 得到任意时间步的 $x_t$：

$$x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon$$