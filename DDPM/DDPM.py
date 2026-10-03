import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

image = imageio.imread("images/img.jpg")
x0 = torch.FloatTensor(image)/255
plt.imshow(x0)

beta_start = 1e-4
beta_end = 0.01
steps = 10
betas = torch.linspace(beta_start,beta_end,10)
alphas = 1-betas
alphas_cumprod = torch.cumprod(alphas,dim = 0)
alphas_cumprod_sqrt1 = torch.sqrt(alphas_cumprod)
alphas_cumprod_sqrt2 = torch.sqrt(1-alphas_cumprod)

#前向反馈加噪
def feedforward(x_condition,t,return_noise = False):
    mean = x_condition*alphas_cumprod_sqrt1[t]
    std = alphas_cumprod_sqrt2[t]
    noise = torch.randn_like(x_condition)
    if return_noise == False:
        return mean + std*noise
    else:
        return mean + std*noise,noise

plt.figure(figsize=(15,5))
show_steps = [0,3,6,9]
for i,step in enumerate(show_steps):
    plt.subplot(1,len(show_steps),i+1)
    if step == 0 : 
        xt = x0
    else:
        xt = feedforward(x0,step,return_noise=False)
    xt = torch.clamp(xt,0,1)
    plt.imshow(xt)
    plt.axis("off")
    plt.title(f"step_{step}")
plt.tight_layout()
plt.show()

#反向去噪
def reverse(x_destination,t,noise):
    x_condition = x_destination/alphas_cumprod_sqrt1[t] - noise*alphas_cumprod_sqrt2[t]/alphas_cumprod_sqrt1[t]
    return x_condition

def Network(x_destination,t,noise):
    return noise

#我们这里模拟noise就是原来的噪声(实际Diffusion Model中采用神经网络进行noise的预测)
t = np.random.randint(1,steps)
xt,noise = feedforward(x_condition=x0,t=t,return_noise=True)

noise = Network(x_destination=xt,t=t,noise=noise)

x0_pre = reverse(xt,t,noise)
plt.figure(figsize=(15,5))
xt = torch.clamp(xt,0,1)
x0 = torch.clamp(x0,0,1)
plt.subplot(1,3,1)
plt.imshow(x0)
plt.axis("off")
plt.title("Original")

plt.subplot(1,3,2)
plt.imshow(xt)
plt.axis("off")
plt.title("Noise")

plt.subplot(1,3,3)
plt.imshow(x0_pre)
plt.axis("off")
plt.title("Prediction")

plt.show()





