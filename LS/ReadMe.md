[TOC] 

最小二乘法(Least Square, LS)

***

# 概述

## 最小二乘思想

​	残差(观测点和估计点的差值)的平方和最小，用于在线性回归模型中估计未知参数的线性最小二乘法



## 推导

​	变量y与x可控变量x~1~ x~2~ ...x~t~之间关系：
$$
y=β_1x+β_2x+...+β_tx+ε
$$
​    转成矩阵格式

$$
V=Ax-B
$$

​	式中,
$$
A=\left[
\matrix{
  β_1^1 & β_2^1 & ... β_k^1\\
  β_1^2 & β_2^2 & ... β_k^2\\
              ...\\
  β_1^n & β_2^n & ... β_k^n\\
  }
\right]
$$

$$
B=\left[
\matrix{
  y^1-ε^1\\
  y^2-ε^2\\
    ...\\
  y^n-ε^n\\
}
\right]
$$

​	转化成最小二乘形式
$$
min||V||
$$
​	求欧几里得空间的2范数作为距离
$$
||V||_2^2= ||Ax-B||_2^2\\
=(Ax-B)^T(Ax-B)\\
=x^TA^TAx-B^TAx-x^TA^TB+B^TB
$$
​	导数为0，求最小值
$$
\frac{∂||Ax-B||_2^2}{∂x}=2A^TAx-2A^TB=0
$$
​	最小二乘的解为：
$$
x=(A^TA)^{-1}(A^TB)
$$
​	加权最小二乘(WLS)
$$
x_{估计}=(A^TPA)^{-1}(A^TPB)
$$

​	模型误差为(n为条件数,t为未知数,分别为矩阵A的行数的列数)
$$
σ=\sqrt(\frac{||Ax_{估计}-B||_2^2}{n-t})
$$


## 特点

* 线性可以按照上式求解，非线性需要迭代
* 矩阵A的行数要大于列数



# 应用