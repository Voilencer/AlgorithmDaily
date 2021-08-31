[TOC]

线性回归

***



# 单变量线性回归

## 模型

$$
h_θ(x) = θ_0 + θ_1x
$$

## 代价函数

$$
J(θ_0, θ_1) = \frac{1}{2m}\sum_{i=1}^m(h_θ(x^{(i)})-y^{(i)})^2
$$



## 梯度下降优化

Goal:
$$
minimizeJ(θ_0, θ_1)
$$
Gradient descent:
$$
θ_j := θ_j - α\frac{∂}{∂θ_j}J(θ_0, θ_1)\\
=θ_j-α\frac{1}{m}\sum_{i=1}^m((h_θ(x^{(i)}) - y^{(i)})·x^{(i)})
$$


# 多变量线性回归

## 模型

$$
h_θ(x) =θ_0x_0 + θ_1x_1 + θ_2x_2 + ... + θ_nx_n\\
=\vec{θ^T}X
$$

## 代价函数

$$
J(θ_0, θ_1) = \frac{1}{2m}\sum_{i=1}^m(h_θ(x^{(i)})-y^{(i)})^2
$$



## 梯度下降优化

Goal:
$$
minimizeJ(θ_0, θ_1)
$$
Gradient descent:
$$
θ_j := θ_j - α\frac{∂}{∂θ_j}J(θ_0, θ_1)\\
=θ_j-α\frac{1}{m}\sum_{i=1}^m((h_θ(x^{(i)}) - y^{(i)})·x^{(i)})
$$


## 特征缩放

​		多维特征都具有相近的尺度，梯度下降算法更快地收敛

​		特征归一化，所有的特征尺度缩放到0到1之间
$$
x_n = \frac{x_n-μ_n}{s_n}
$$


## 正规方程

$$
h_θ(x) =θ_0x_0 + θ_1x_1 + θ_2x_2 + ... + θ_nx_n\\
= \vec{X}\vec{θ}
$$

​		最小二乘法求解
$$
θ = (X^TX)^{-1}X^Th
$$
​		$X^TX$如果不可逆，需要使广义逆



# 正则化线性回归

特征太多，数据量太少，过度拟合就会发生

解决过度拟合方法：

1. 减少特征数量：	1）手动选择；	2）优化；

2. 正则化：保留所有的参数，减少参数的大小（参数前系数变大）

## 梯度下降法优化

​		代价函数
$$
J\left( \theta  \right)=\frac{1}{2m}[\sum\limits_{i=1}^{m}{{{({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})}^{2}}+\lambda \sum\limits_{j=1}^{n}{\theta_{j}^{2}}]}
$$
​		梯度下降法优化（$j≥1$）
$$
{\theta_j}:={\theta_j}(1-a\frac{\lambda }{m})-a\frac{1}{m}\sum\limits_{i=1}^{m}{({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})x_{j}^{\left( i \right)}}
$$

## 正规方程优化

![image-20210325083714232](C:\Users\Voilencer\AppData\Roaming\Typora\typora-user-images\image-20210325083714232.png)