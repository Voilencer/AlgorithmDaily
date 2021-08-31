[TOC]

非极大值抑制(Non-Maximum Suppression, NMS)

***

# 概述

## 简介

​	目标检测得到一系列冗余的检测框，抑制不是极大值的元素，可以理解为局部最大搜索，保留最好的检测结果。

## 原理

​	遍历将所有的框得分排序，选中其中得分最高的框，然后遍历其余框找到和当前最高分的框的重叠面积（IOU）大于一定阈值的框，删除。然后继续这个过程，找另一个得分高的框，再删除IOU大于阈值的框，循环。

## 步骤

（1）假设有6个矩形框，根据分类器的类别分类概率做排序，假设从小到大属于车辆的概率 分别为A、B、C、D、E、F；

（2）从最大概率矩形框F开始，分别判断A~E与F的重叠度IOU是否大于某个设定的阈值；
$$
s_i=\begin{cases}
		s_i,  iou < thr\\
		0,    iou >= thr
     \end{cases}
$$
（3）假设B、D与F的重叠度超过阈值，那么就扔掉B、D；并标记保留第一个矩形框F；

（4）从剩下的矩形框A、C、E中，选择概率最大的E，然后判断E与A、C的重叠度，重叠度大于一定的阈值，那么就扔掉；并标记E是保留下来的第二个矩形框；

（5）重复以上步骤，直到找到所有矩形框；



## soft-nms

​	NMS在面对目标密集，或者遮挡时，存在漏检情况。

​	soft-NMS指定了一个置信度评价函数，综合考虑置信度和IOU来判断得分，大于指定阈值的仍然保留。

​	两种改进方式：

（1）线性加权
$$
s_i=\begin{cases}
		s_i,  iou < thr\\
		s_i(1-IOU),    iou >= thr
     \end{cases}
$$
（2）高斯加权
$$
s_i=\begin{cases}
		s_i,  iou < thr\\
		s_ie^{-\frac{IOU^2}{σ}},    iou >= thr
     \end{cases}
$$




## 对比

|              |                                                              |
| ------------ | ------------------------------------------------------------ |
| NMS          | 目标密集或者遮挡时，存在漏检情况                             |
| soft-nms     | 综合考虑IOU和目标置信度，根据得分判断是否保留<br />需要手动设置阈值 |
| adaptive-nms | 根据目标密度自动设置阈值                                     |
| INMS         | 计算rbox的IOU                                                |
| PNMS         | 计算多边形的IOU                                              |
| MNMS         | 基于分割区域的IOU                                            |



# 应用

* 目标检测