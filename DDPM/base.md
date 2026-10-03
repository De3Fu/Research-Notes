# 1. Denoising Diffusion Probabilistic Model

**Denoising Diffusion Probabilistic Model（DDPM，去噪扩散概率模型）**主要包含两个过程：

- Forward Process：前向加噪
- Reverse Process：反向去噪 / 生成

---

## 1.1 Forward Diffusion Process

DDPM 的前向过程会不断向原始图像中添加高斯噪声。

首先定义：

$$\alpha_t = 1-\beta_t$$

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

其中：

$$\epsilon\sim\mathcal{N}(0,I)$$

因此，可以把 $x_t$ 理解为两部分：

$$x_t=\underbrace{\sqrt{\bar{\alpha}_t}x_0}_{\text{Image Signal}}+\underbrace{\sqrt{1-\bar{\alpha}_t}\epsilon}_{\text{Noise}}$$

随着 $t$ 不断增大：

$$\bar{\alpha}_t\rightarrow 0$$

原始图像的信息越来越少，而噪声越来越多。

最终：

$$x_T\approx\mathcal{N}(0,I)$$

即接近标准高斯噪声。

---

# 2. Noise Schedule

$\beta_t$ 可以理解为噪声调度中的基本参数，它决定每一个时间步加入多少噪声。

通常：

$$0<\beta_1<\beta_2<\cdots<\beta_T<1$$

随着时间步增加，图像中的累计噪声越来越多。

---

## 2.1 Linear Schedule

一种简单的方法是让 $\beta_t$ 线性增加：

$$\beta_t=\beta_{\min}+\frac{t}{T}\left(\beta_{\max}-\beta_{\min}\right)$$

例如：

$$\beta_{\min}=10^{-4}$$

$$\beta_{\max}=0.02$$

这样随着 $t$ 增大，每一步加入的噪声强度逐渐发生变化。

---

## 2.2 Cosine Schedule

除了 Linear Schedule，还可以采用 **Cosine Schedule**。

Cosine Schedule 通常不是简单地直接令 $\beta_t$ 按余弦变化，而是首先设计累计信号量：

$$\bar{\alpha}_t$$

使 $\bar{\alpha}_t$ 按照余弦形式变化，再由 $\bar{\alpha}_t$ 推导对应的 $\beta_t$。

这种方式能够使整个扩散过程中信号的衰减更加平滑。

---

# 3. Signal-to-Noise Ratio

对于：

$$x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon$$

可以看到，原始信号对应的方差权重为：

$$\bar{\alpha}_t$$

噪声对应的方差权重为：

$$1-\bar{\alpha}_t$$

因此可以定义信噪比：

$$\mathrm{SNR}(t)=\frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}$$

SNR 可以直观理解为：

> 当前 $x_t$ 中，原始图像信号相对于噪声还剩多少。

在**前向扩散过程**中：

$$t\uparrow\quad\Rightarrow\quad\bar{\alpha}_t\downarrow\quad\Rightarrow\quad\mathrm{SNR}(t)\downarrow$$

即随着不断添加噪声，图像信号越来越弱。

而在**反向去噪过程**中：

$$x_T\rightarrow x_{T-1}\rightarrow\cdots\rightarrow x_0$$

随着去噪不断进行，SNR 会逐渐增大。

---

# 4. Reverse Denoising Process

前向过程最终把图像逐渐变成接近高斯噪声的状态：

$$x_T\approx\mathcal{N}(0,I)$$

生成过程则从随机高斯噪声开始：

$$x_T\sim\mathcal{N}(0,I)$$

然后进行：

$$x_T\rightarrow x_{T-1}\rightarrow\cdots\rightarrow x_1\rightarrow x_0$$

逐步恢复出图像。

---

## 4.1 Predicting Noise

经典 DDPM 中，神经网络通常不直接预测 $x_0$，而是预测当前 $x_t$ 中包含的噪声：

$$\hat{\epsilon}=\epsilon_\theta(x_t,t)$$

在文本条件扩散模型中，还可以加入文本条件 $c$：

$$\hat{\epsilon}=\epsilon_\theta(x_t,t,c)$$

其中：

- $x_t$：当前时间步的噪声图像
- $t$：当前时间步
- $c$：文本或其他条件
- $\epsilon_\theta$：神经网络
- $\hat{\epsilon}$：模型预测出来的噪声

---

## 4.2 从预测噪声估计 $x_0$

根据前向公式：

$$x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon$$

可以反推出：

$$\hat{x}_0=\frac{x_t-\sqrt{1-\bar{\alpha}_t}\hat{\epsilon}}{\sqrt{\bar{\alpha}_t}}$$

也可以写成：

$$\hat{x}_0=\frac{1}{\sqrt{\bar{\alpha}_t}}x_t-\sqrt{\frac{1-\bar{\alpha}_t}{\bar{\alpha}_t}}\hat{\epsilon}$$

但在真正的 DDPM Sampling 中，并不是每一步直接得到最终的 $x_0$。

模型会根据预测出的噪声构造反向分布：

$$p_\theta(x_{t-1}\mid x_t)$$

然后逐步进行：

$$x_t\rightarrow x_{t-1}$$

最终经过多步去噪得到 $x_0$。

---

# 5. DDPM Training Process

DDPM 的一次训练过程可以概括为以下六个步骤。

### Step 1：从数据集中采样真实图像

$$x_0\sim p_{\text{data}}(x)$$

### Step 2：随机采样时间步

$$t\sim\mathrm{Uniform}\{1,\ldots,T\}$$

### Step 3：采样高斯噪声

$$\epsilon\sim\mathcal{N}(0,I)$$

### Step 4：直接构造对应的 $x_t$

$$x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon$$

注意，这里不需要真的从 $x_0$ 一步一步加噪到 $x_t$。

由于前向过程具有闭式表达，我们可以随机采样任意一个时间步 $t$，直接构造对应的 $x_t$。

### Step 5：神经网络预测噪声

无条件 DDPM：

$$\hat{\epsilon}=\epsilon_\theta(x_t,t)$$

条件扩散模型：

$$\hat{\epsilon}=\epsilon_\theta(x_t,t,c)$$

### Step 6：计算 Noise Prediction Loss

经典 DDPM 常用的简化训练目标为：

$$\mathcal{L}_{\text{simple}}=\mathbb{E}_{x_0,t,\epsilon}\left[\left\|\epsilon-\epsilon_\theta(x_t,t,c)\right\|_2^2\right]$$

其核心思想就是：

> **真实加入了什么噪声，就训练神经网络把这个噪声预测出来。**

因此，DDPM 最核心的训练逻辑可以概括为：

$$x_0\rightarrow\text{random }t\rightarrow\text{add noise}\rightarrow x_t\rightarrow\epsilon_\theta\rightarrow\hat{\epsilon}$$

训练的目标就是让：

$$\hat{\epsilon}\approx\epsilon$$

---

# 6. 核心理解

DDPM 的训练本质上是在训练一个神经网络：

> 给我一个被噪声污染到某种程度的图像 $x_t$，再告诉我当前时间步 $t$，模型需要判断这个 $x_t$ 中包含了什么噪声。

训练完成之后，我们就可以从纯高斯噪声：

$$x_T\sim\mathcal{N}(0,I)$$

开始不断预测并去除噪声：

$$x_T\rightarrow x_{T-1}\rightarrow\cdots\rightarrow x_0$$

最终生成一个符合数据分布的样本。