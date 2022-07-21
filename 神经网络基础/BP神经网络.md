# BP神经网络

## 学习法则

### 赫布学习法则

∆$\omega$=$\varphi y_i y_j$ 

其中$\Delta \omega$ 表示为连接强度（权重）的变化量，$\varphi$  为一常量， $y_i$ 表示为突触前膜神经元的兴奋程度,$y_j$ 表示为突触后膜神经元的兴奋程度。

意义：突触前膜和突触后膜产生的兴奋使突触的能量传递效率增强。与之相反，长时间没有兴奋，突触的能量传递效率衰退。

### Delta学习法则

$\Delta\omega=\eta(y_j-t)y_i$ 

其中$\Delta \omega$ 表示为权重的变化量，$y_i$ 突触的前神经元的输出，$y_j$ 表示突触的后神经元的输出，t表示正确答案，$\eta$表示学习系数的常数。

意义：

* 如果输出与正确答案之间的差值越大，则需要设置的权重的修正量也越大。
* 如果输入越大，则需要设置的权重的修正量也越大。

## 反向传播

* 训练数据与测试数据
* 损失函数
* 梯度下降法
* 最优化算法
* 批次尺寸

### 训练数据与测试数据

训练数据：在神经网络的学习中使用的数据

测试数据：对学习结果进行验证使用的数据

样本：由一对输入数据和正确答案构成的组合称为样本。

正确答案：

回归问题：正确答案可使用并列的数值所组成的向量

分类问题：正确答案可以使用只用一个元素值是1，而其他为0的向量表示。

这种只有一个1而其他都为0所并列组成数值的格式成为`独热编码`。

### 损失函数

对输出与正确答案的误差进行定义的函数就是损失函数

#### 平方和误差（回归问题）

$E=\frac{1}{2}\sum(y_k-t_k)^2$

其中 E 表示误差，$y_k$表示输出层的各个输出值，$t_k$表示正确答案

```python
import numpy as np
def square_sum(y,t):
    1.0/2.0*np.sum(np.square(y-t))
```

#### 交叉熵误差(分类问题)

$E=-\sum t_klog(y_k)$

其中 E 表示误差，$t_k$表示正确答案，$y_k$表示输出答案，由于$t_k$是独热编码，故只有唯一正确为1的项对误差产生影响。

```python
import numpy as np
def cross_entropy(y,t):
	return -np.sum(t*np.log(y+1e-7))
```

### 梯度下降法

正向传播，对网络层的输出进行传播，反向传播，对输入的梯度进行传播。

输出层：

* 偏置的梯度

* 权重的梯度

* 输入的梯度

中间层：

* 偏置的梯度
* 权重的梯度
* 输入的梯度

| 网络层  | 下表   | 神经元数量 |
| ---- | ---- | :---- |
| 输入层  | i    | l     |
| 中间层  | j    | m     |
| 输出层  | k    | n     |

#### 输出层的梯度

##### 相关函数

$ E=f_1(y_k)$

$y_k=f_2(u_k)$

$u_k=\sum_{q=1}^{m}(y_qw_{qk}+b_k)$

E:损失值

$f_1$:损失函数

$y_k$:输出结果

$f_2$:激励函数

$u_k$:输入与权值乘积与偏置的求和

$y_j$:$y_j$为$y_q$中的一个

##### 定义

$\delta_k=\frac{∂E}{∂u_K}=\frac{\partial E}{\partial y_k}\frac{\partial y_k}{\partial u_k}$

$\frac{\partial E}{\partial y_k}$损失函数对$y_k$的偏微分

$\frac{\partial y_k}{\partial u_k}$激励函数对$u_k$的偏微分

##### 输出层权重的梯度

$\partial w_{jk} = \frac{\partial E}{\partial w_{jk}}$

根据微分连锁公式

$\frac{\partial E}{\partial w_{jk}}=\frac{\partial E}{\partial y_k}\frac{\partial y_k}{\partial u_k}\frac{\partial u_k}{\partial w_{jk}}$

$\frac{\partial u_k}{\partial w_{jk}}=y_j​$   :$u_k​$对$w_{jk}​$的偏微分

$\partial w_{jk} =\delta_ky_j$

##### 输出层偏置的梯度

$\partial b_k = \frac{\partial E}{\partial b_k}$

根据微分连锁公式

$\frac{\partial E}{\partial b_k}=\frac{\partial E}{\partial y_k}\frac{\partial y_k}{\partial u_k}\frac{\partial u_k}{\partial b_k}$

$\frac{\partial u_k}{\partial b_k}=1$  : $u_k$对$b_k$的偏微分

$\partial b_k =\delta_k$

#### 输出层的输入梯度

输出层的输入梯度=中间层的输出梯度

$\partial y_j= \frac{\partial E}{\partial y_j}$

根据微分连锁公式

$\frac{\partial E}{\partial y_j}=\sum_{r=1}^{n}\frac{\partial E}{\partial y_r}\frac{\partial y_r}{\partial u_r}\frac{\partial u_r}{\partial y_j}$

$\frac{\partial u_r}{\partial y_j}=w_{jr}$

$\partial y_j=\sum_{r=1}^{n}\delta_rw_{jr}$

#### 中间层的梯度

##### 相关函数

$y_j=f(u_j)$

$u_j=w_{ij}y_i+b_j$

$y_j$为中间层的输出值

$f$为激励函数

$u_j$输入值与权重的乘积与偏置的和

##### 中间层权重的梯度

$∂ w_{ij}=\frac{∂ E}{∂ w_{ij}}=\frac{∂ E}{∂ uj}\frac{∂{u_j}}{∂ w_{ij}}=\frac{∂ E}{∂ y_j}\frac{∂ y_j}{∂ u_j}\frac{∂ u_j}{∂ w_{ij}}$

$\frac{∂ E}{∂ y_j}$为中间层的输出梯度

$\frac{∂ y_j}{∂ u_j}$为激励函数的微分

$\frac{∂ u_j}{∂ w_{ij}}=y_i$

定义$\delta_j=∂ y_i\frac{∂ y_j}{∂ u_j}$则$∂ w_{ij}=yi\delta_j$

##### 中间层偏置的梯度

$∂ b_j=\frac{∂ E}{∂ b_j}=\frac{∂ E}{∂ uj}\frac{∂{u_j}}{∂ b_j}$

$\frac{∂{u_j}}{∂ b_j}=1$

则$ b_j=\delta_j$

若网络层不止三层，即隐藏层不止三层，则$∂ y_i=\sum_{q=1}^{m}∂_qw_{iq}$

#### 梯度计算公式总结

##### 输出层

$\delta_k=\frac{\partial E}{\partial u_k}=\frac{\partial E}{\partial y_k}\frac{\partial y_k}{\partial u_k}$

$\partial w_{jk} =y_j\delta_k$

$\partial b_k =\delta_k$

$\partial y_j=\sum_{r=1}^{n}\delta_rw_{jr}$

关于$\delta_k$的求解，在使用不同的损失函数和激励函数组合时不同，其方法也是不同的，$\delta_k$与输出层的神经元数量相同

##### 中间层

$\delta_j=\frac{∂ E}{∂ u_j}=∂ y_i\frac{∂ y_j}{∂ u_j}$

$∂ w_{ij}=y_i\delta_j$

$∂ b_j=\delta_j$

$∂ y_i=\sum_{q=1}^{m}∂_qw_{iq}$

### 最优化算法

* 随机梯度下降法(SGD)
* Momentum
* AdaGrad
* RMSProp
* Adam

#### 随机下降梯度法(SGD)

更新公式：
$w\leftarrow w-\eta\frac{∂ E}{∂ w}$

$b\leftarrow b-\eta\frac{∂ E}{∂ b}$

优点：随机选样本，不容易掉入局部最优解。简单确定更新量，简单的代码实现。

缺点：在学习的过程中无法对更新量灵活调整

#### Momentum

$w\leftarrow w-\eta\frac{∂ E}{∂ w}+\alpha\Delta w$

$b\leftarrow b-\eta\frac{∂ E}{∂ b}+\alpha\Delta b$

$\alpha$决定惯性的强度常量，$\Delta w$表示前一次的更新量

优点：防止更新量的急剧变化

缺点：必须事先给定$\eta$和 $\alpha$，增加网络调整的难度。

#### AdaFrad

$ h\leftarrow h+(\frac{∂ E}{∂ w})^2$

$ w\leftarrow w-\eta\frac{1}{\sqrt{h}}\frac{∂ E}{∂ w}$

$ h\leftarrow h+(\frac{∂ E}{∂ b})^2$

$ b\leftarrow b-\eta\frac{1}{\sqrt{h}}\frac{∂ E}{∂ b}$

优点：对更新量进行调整

缺点：更新量持续减少

#### RMSPorp

$ h\leftarrow \rho h+(1-\rho)(\frac{∂ E}{∂ w})^2$

$ w\leftarrow w-\eta\frac{1}{\sqrt{h}}\frac{∂ E}{∂ w}$

$ h\leftarrow \rho h+(1-\rho)(\frac{∂ E}{∂ b})^2$

$ b\leftarrow b-\eta\frac{1}{\sqrt{h}}\frac{∂ E}{∂ b}$

$\rho$一般设置为0.9。

#### Adam

$m_0=v_0=0$

$ m_t=\beta_1m_{t-1}+(1-\beta_1)\frac{∂ E}{∂ w}$

$ v_t=\beta_2v_{t-1}+(1-\beta_2)(\frac{∂ E}{∂ w})^2$

$\hat{m_t}=\frac{m_{0t}}{1-\beta_1^t}$

$\hat{v_t}=\frac{v_t}{1-\beta_2^t}$

$w \leftarrow w-\eta\frac{\hat{m_t}}{\sqrt{\hat{v_t}}+\omicron}$

$\beta_1=0.9 \ \ \beta_2=0.999\ \ \eta=0.001\ \ \ \omicron=10^{-8}  $

t表示重复次数

### 批次尺寸

epoch：完成一次对所有训练数据的学习被称为一轮epoch

样本：实际进行学习时，将多个训练数据的样本集合在一起进行学习的，这些样本的集合被称为批次。

批次尺寸：一轮epoch中使用的训练数据可以分割成多个批次学习。一个批次中包含样本的数量被称为批次尺寸。

#### 批次学习

批次学习：批次尺寸是全部训练数据。

$E=\frac{1}{N}\sum_{i=1}^{N}E_i$

$ \frac{∂ E}{∂ w}=\sum_{i=1}^{N}\frac{∂ E_i}{∂ w}$

N:训练数据的个数

$E_i$表示每个数据的误差

#### 在线学习

在线学习：批次尺寸为1的学习

#### 小批次学习

小批次学习：位于批次学习和在线学习之间就是小批次学习。

$E=\frac{1}{n}\sum_{i=1}^{n}E_i$

$ \frac{∂ E}{∂ w}=\sum_{i=1}^{n}\frac{∂ E_i}{∂ w}$

n:训练数据的个数(n<N)

### 矩阵运算

输入：批次尺寸×传递到网络层的输入尺寸

输出：批次尺寸×网络层的输出数量（网络层的神经元数量）

正确答案：批次尺寸×正确答案的数量

矩阵的每一行与每个样本对应的

#### 使用矩阵进行正向传播

假设现在在一个神经元数量为n的网络层这进行正向传播，批次尺寸用h表示，输入数据的数量（位于上层网络的神经元数量）用m表示，则用于表示输入的矩阵X的尺寸为$h×m$，权重的矩阵W尺寸为$m×n$。偏置$\overrightarrow{b}$为向量，利用广播效应得到$U=XW+\overrightarrow{b}$

再利用激励函数$f$之后得到结果Y。

$X =\begin{pmatrix}X_{11}\ \ X_{12}\ \ ...\ X_{1m}\\X_{21}\ \ X_{22}\ \ ...\ X_{2m}\\.............\\X_{h1}\ \ X_{h2}\ \ ...\ X_{hm} \end{pmatrix}$

#### 使用矩阵进行反向传播

$\Delta=\begin{pmatrix}\delta_{11}\ \ \delta_{12}\ \ ...\ \delta_{1n}\\\delta_{21}\ \ \delta_{22}\ \ ...\ \delta_{2n}\\.............\\\delta_{h1}\ \ \delta_{h2}\ \ ...\ \delta_{hn} \end{pmatrix}$

对于权重的梯度为$\delta w_{ij}=\frac{\delta E}{\delta w_{ij}}=y_i\delta_j$

由于批次学习，则为$\sum_{k=1}^{h}\frac{\delta E_k}{\delta w_{ij}}$

则$ ∂ W=X^{T}\Delta\\ =\begin{pmatrix}X_{11}\ \ X_{21}\ \ ...\ X_{h1}\\X_{12}\ \ X_{22}\ \ ...\ X_{h2}\\.............\\X_{1m}\ \ X_{2m}\ \ ...\ X_{hm} \end{pmatrix}\begin{pmatrix}\delta_{11}\ \ \delta_{12}\ \ ...\ \delta_{1n}\\\delta_{21}\ \ \delta_{22}\ \ ...\ \delta_{2n}\\.............\\\delta_{h1}\ \ \delta_{h2}\ \ ...\ \delta_{hn} \end{pmatrix}$

要进行矩阵乘法运算$X$必须转置。

```python
grad_w=np.dot(x.T,delta)
```

偏置的梯度公式：$∂  b_j=∂_j$由于批次学习，则为$\sum_{k=1}^{h}\frac{∂ E_k}{∂b_j}$

```python
grad_b=np.sum(delta,axis=0)
```

上层网络的输出梯度为$∂ y_j=\sum_{r=1}^{n}\delta_rw_{jr}$

$∂ X=\Delta W^{T}\\=\begin{pmatrix}\delta_{11}\ \ \delta_{12}\ \ ...\ \delta_{1n}\\\delta_{21}\ \ \delta_{22}\ \ ...\ \delta_{2n}\\.............\\\delta_{h1}\ \ \delta_{h2}\ \ ...\ \delta_{hn} \end{pmatrix}\begin{pmatrix}w_{11}\ \ w_{21}\ \ ...\ w_{m1}\\w_{12}\ \ w_{22}\ \ ...\ w_{m2}\\.............\\w_{1n}\ \ w_{2n}\ \ ...\ w_{mn} \end{pmatrix}$



```python
grad_x=np.dot(delta,w.T)
```

