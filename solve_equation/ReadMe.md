[TOC]

齐次方程组和非齐次方程组求解

***

* 秩：矩阵的非零子式的最高阶数
* 增广矩阵：在系数矩阵的右边添上一列，这一列是线性方程组的等号右边的值
* 超定方程：有效方程个数大于未知数个数的方程组，m > n
* 负定/欠定：有效方程个数小于未知数个数的方程组

# Ax=b

* A是m x n矩阵
* 常数项不全为零的线性方程组称为非齐次线性方程组

## 解分析

| 秩           | 解分析                                                       |
| ------------ | ------------------------------------------------------------ |
| r = m = n    | 行满秩和列满秩， 方阵，有唯一解 ，                           |
| r = n <m     | 列满秩， 0解或者1个解（b如果是A的列的线性组合有唯一解）      |
| r = m < n    | 行满秩，无穷多个解                                           |
| r < m, r < n | 0或者无穷多个解（b的行和A的行向量之间有相同的组合关系，则有无穷解，否则0解） |

## 求解

$$
\left[ \begin{matrix} 
1 & 2 & 2 & 2 \\ 
2 & 4 & 6 & 8 \\ 
3 & 6 & 8 & 10 
\end{matrix} \right]
\left[\begin{matrix}
x_1 \\ x_2 \\ x_3 \\ x_4
\end{matrix}\right] 
=
\left[\begin{matrix}
b_1 \\ b_2 \\ b_3
\end{matrix}\right]
$$

### 矩阵变换

* 构建增广矩阵$[A|b]$

$$
\left[ \begin{matrix} 
1 & 2 & 2 & 2 & b_1\\ 
2 & 4 & 6 & 8 & b_2\\ 
3 & 6 & 8 & 10 & b_3 
\end{matrix} \right]
$$

* 对增广矩阵初等行变换

$$
\left[ \begin{matrix} 
1 & 2 & 2 & 2 & b_1\\ 
0 & 0 & 2 & 4 & b_2-2b_1\\ 
0 & 0 & 0 & 0 & b_3-b_2-b_1 
\end{matrix} \right]
$$

* 分析矩阵A的秩，见上表
  * $r(A)< 3, r(A)<4$，  解有0个或无穷个

* 分析$b$ 是否和A的行有相似线性关系
  * $b_3-b_2-b_1 ≠ 0$，0解
  *  $b_3-b_2-b_1 = 0$，无穷解



### SVD分解

* 矩阵A进行SVD分解
* 最优化问题

$$
argmin||Ax - b|| = argmin||UDV^Tx - b || = argmin||DV^Tx - U^Tb||
$$

* 令$y = V^Tx$ ， $b' = U^Tb$

$$
min(||DV^Tx - U^Tb||_2) = min(||D y -b'||_2)
$$

* 求解向量y

$$
y = D^{-1}b'
$$

* 求解向量x

$$
x = Vy
$$



### 最小二乘解

$\hat{x} = (A^TA)^{-1}(A^Tb)$

* $A^TA$可逆，非奇异



# Ax = 0

* A是m x n矩阵

## 解分析

* 存在精确解的条件是： r < n

## 求解

### SVD分解

* 正交矩阵的保范性

$$
||UDV^Tx|| = ||DV^Tx||
$$

* SVD分解

$$
A = UDV^T
$$

D为对角矩阵,U,V分别为正交矩阵

* 目标最小

$$
argmin||Ax|| = argmin||UDV^Tx|| = argmin||DV^Tx||\\
$$

* 令$y = V^Tx$

$$
argmin||Dy||
$$

* $||y||=1$时，最小化$||Dy||$
* 矩阵D是由A矩阵SVD分解的特征值组成的对角矩阵，对角元素降序排列，最优解在$y = (0,0,...,1)^T$取得
* $x = Vy$， 最小特征值对应的特征向量是超定方程的解

### 特征值和特征向量

* $argmin||Ax||$

$$
||Ax|| = (Ax)^T(Ax) = x^TA^TAx = x^T(A^TA)x
$$

* 令$A^TAx = λx$， `λ` 和`x`是$A^TA$ 的特征值和特征向量

$$
||Ax|| = x^Tλx = λx^Tx
$$

* 因为$Ax=0$， $x$是方程的一个解，可以限制$||x|| = x^Tx = 1$

$$
||Ax|| =  λ
$$

* 所以$λ$最小时, 方程$argmin||Ax||$
* 所以超定方程$Ax=0$的解就是$A^TA$的最小特征值所对应的特征向量



# python求解

* SVD 

```
def solve1(A):
    # A SVD分解，最小特征值对应的特征向量, V最后一列,Vt最后一行
    A = np.matrix(A)
    Q, S, Vt = np.linalg.svd(A)
    return Vt[-1, :]

if __name__ == "__main__":
    A = np.array([[4,5,6], [1,2,3]])
    res1 = solve1(A)
    tmp1 = np.matrix(A) * np.matrix(res1).T
    print(tmp1)
```



* 特征值、向量

```
def solve(A):
    # AtA最小特征值对应特征向量
    A = np.mat(A)
    U = A.T * A
    lamda, hU = np.linalg.eig(U)
    return hU[:, -1]
    
if __name__ == "__main__":
    A = np.array([[4,5,6], [1,2,3]])
    res2 = solve(A)
    tmp2 = np.matrix(A) * np.matrix(res2)
    print(tmp2)    
```