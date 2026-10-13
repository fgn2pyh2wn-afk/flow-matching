[Flow_Matching.md](https://github.com/user-attachments/files/33361293/Flow_Matching.md)
# Flow Matching 学习笔记

> 配套实验：`Diffusion_FlowMatching_MNIST.ipynb`（MNIST，PyTorch + MPS）
> 一句话概括：**把生成问题变成"学习一个从噪声指向数据的速度场"，然后沿这个速度场解一条 ODE。**

---

## 1. 动机：为什么需要 Flow Matching

扩散模型（DDPM）的思路是"先加噪、再去噪"，其反向过程是一条弯弯曲曲的路径，采样往往需要几百上千步。

Flow Matching 的思路是：**干脆直接定义一条从噪声 $x_1$ 到数据 $x_0$ 的路径，学习这条路径上的速度场**，生成时从噪声出发沿速度场"一路走到底"。路径越直，需要的步数越少。

| | 扩散模型（DDPM） | Flow Matching |
| --- | --- | --- |
| 前向过程 | 随机加噪：$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$ | 确定性插值：$x_t=(1-t)x_0+t\,x_1$ |
| 学习目标 | 预测噪声 $\epsilon$ | 预测速度 $v=x_1-x_0$ |
| 生成方向 | 从 $t=T$ 到 $t=0$ 逐步"去噪" | 从 $t=1$ 到 $t=0$ 沿速度场"前进" |
| 采样器 | DDIM / DDPM（反向 ODE / 随机链） | Euler 等 ODE 求解器 |

---

## 2. 数学设定

- 数据：$x_0 \sim p_{data}$（例如 MNIST 图片）
- 噪声源：$x_1 \sim \mathcal{N}(0, I)$
- **直线插值路径**：

$$
x_t = (1-t)\,x_0 + t\,x_1, \qquad t\in[0,1]
$$

对 $t$ 求导就得到这条路径的速度：

$$
\frac{dx_t}{dt} = x_1 - x_0
$$

注意：**速度是常数**（与 $t$ 无关）——这正是"直线路径"的好处，目标函数变得极其简单。

---

## 3. Conditional Flow Matching（CFM）损失

我们希望网络 $v_\theta(x_t, t)$ 输出的速度场逼近真实速度：

$$
\mathcal{L} = \mathbb{E}_{t\sim U[0,1],\ x_0\sim p_{data},\ x_1\sim\mathcal N(0,I)}
\big\| v_\theta(x_t, t) - (x_1 - x_0) \big\|^2
$$

相比最早的连续归一化流（CNF）需要数值解 ODE、算对数密度，CFM 把训练简化成了**一次采样 + 一个 MSE**，写起来和 DDPM 一样简单。

---

## 4. 训练算法（伪代码）

```
repeat:
    x0 ~ 训练数据                # 一张真实图片
    x1 ~ N(0, I)                 # 随机噪声
    t  ~ Uniform(0, 1)
    xt = (1 - t) * x0 + t * x1   # 路径上的中间点
    v_target = x1 - x0           # 真实速度
    loss = MSE(v_theta(xt, t), v_target)
    反向传播更新 v_theta
```

---

## 5. 采样（生成）算法

从纯噪声出发，对 ODE $\dot{x} = v_\theta(x, t)$ 做 Euler 积分，$t$ 从 1 积到 0：

```
x ~ N(0, I)                      # 起点是纯噪声
for t in [1, 0) 逐小步 dt 递减:
    x = x + v_theta(x, t) * dt   # 沿速度场前进一小步
return x                          # t=0 时的 x 就是生成结果
```

---

## 6. 对应 notebook 里的代码

**训练循环**（`FlowMatching` cell）：

```python
t  = torch.rand(b, 1, 1, 1, device=device)  # t ~ U[0,1]
x1 = torch.randn_like(x0)                   # 目标噪声
x_t = (1 - t) * x0 + t * x1                 # 直线插值路径
v_target = x1 - x0                          # 真实速度

v_pred = flow_model(x_t, t.view(b, 1))      # 网络预测速度场
loss = F.mse_loss(v_pred, v_target)
```

**采样**（`flow_sample`）：

```python
@torch.no_grad()
def flow_sample(model, n_samples=64, steps=100):
    x = torch.randn(n_samples, 1, 28, 28, device=device)
    ts = torch.linspace(1.0, 0.0, steps + 1)
    for i in range(steps):
        t  = ts[i]
        dt = ts[i + 1] - ts[i]
        tvec = torch.full((n_samples, 1), t, device=device)
        x = x + model(x, tvec) * dt          # Euler 步
    return x.clamp(0, 1)
```

要点：

- 网络输入是"中间状态 $x_t$ + 时间 $t$"，输出是速度场（形状与图片相同）；
- 时间 $t$ 归一化到 $[0,1]$ 再喂给时间嵌入层；
- 训练和采样里 `t` 的广播形状不同：训练时 `(b,1,1,1)` 才能和图片 `(b,1,28,28)` 对齐相乘。

---

## 7. 与扩散模型训练目标的对应关系

两者其实都是"教网络预测一个量"：

| 方法 | 网络预测 | 训练信号来源 |
| --- | --- | --- |
| Diffusion | 噪声 $\epsilon$ | 前向加噪时真实加入的 $\epsilon$ |
| Flow Matching | 速度 $v=x_1-x_0$ | 路径两端的差（数据与噪声之差） |

实验里两个模型共用**同一个时间条件 U-Net**，只是换了监督目标，最后都能从纯噪声生成数字——说明"预测什么"是可以替换的，而"用时间条件网络 + MSE"这套框架是共通的。

---

## 8. 进阶话题

- **Rectified Flow**：对直线路径做迭代"矫正"（每轮用上一轮生成的轨迹重定义路径），路径越来越直，最终接近最优传输（OT）映射，采样步数可以压到个位数；
- **大模型应用**：Stable Diffusion 3 等新一代模型采用的就是 rectified flow 类目标，这也是"路径更直 → 采样更快"思路的直接受益者；
- **概率解释**：Flow Matching 与扩散的概率流 ODE、score matching 在连续时间框架下相互关联。

---

## 9. 优缺点小结

**优点**

- 训练目标极其简单（一次采样 + MSE），没有密度计算、没有数值 ODE；
- 路径直，采样步数天然可以更少；
- 与最优传输/rectified flow 有清晰的理论联系。

**缺点**

- 基本版（无条件直线插值）在部分基准上质量略逊于精细调度的扩散模型，往往需要矫正/蒸馏来补齐；
- 直线路径隐含"配对"假设，对高维复杂分布的建模能力依赖路径设计。

---

## 10. 实验结论回顾（MNIST）

- Diffusion（DDPM 训练 + DDIM 100 步采样）与 Flow Matching（Euler 100 步）都能从纯高斯噪声生成可辨认数字；
- 两者训练 loss 都收敛，且采样速度一致（都是 100 步 ODE 积分）；
- 两条路线共享同一 U-Net 结构，说明框架是通用的，差异只在"预测目标 + 采样方向"。
