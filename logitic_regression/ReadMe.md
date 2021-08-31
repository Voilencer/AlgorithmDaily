[TOC]

逻辑回归

***

# 逻辑回归

## 模型表示

​		simoid函数：非线性化；（0，1）
$$
h_θ(x) = g(θ^TX)\\
g(z) = \frac{1}{1 + e^{-z}}
$$
![image-20210325083947574](C:\Users\Voilencer\AppData\Roaming\Typora\typora-user-images\image-20210325083947574.png)

## 代价函数

$$
J\left( \theta  \right)=-\frac{1}{m}\sum\limits_{i=1}^{m}{[{{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)+\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]}
$$



## 梯度下降法优化

$$
θ_j := θ_j - α\frac{1}{m}\sum\limits_{i=1}^{m}{[{h_\theta}\left( {{x}^{(i)}} \right)-{{y}^{(i)}}]x_j^{(i)}}
$$



## 多分类

​		训练多个逻辑回归分类器：$h_\theta^{\left( i \right)}\left( x \right)$， 其中 $i$ 对应每一个可能的 $y=i$，

## 正则化逻辑回归

$$
J\left( \theta  \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)-\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]}+\frac{\lambda }{2m}\sum\limits_{j=1}^{n}{\theta _{j}^{2}}
$$

