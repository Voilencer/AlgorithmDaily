import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



"""
GD法求函数极值
f(x) = 3x1^2 + 4x2^2
"""



def fun(x, y):
    return 3 * x**2 + 4 * y**2


def show_func():
    fig1 = plt.figure()  # 创建一个绘图对象
    ax = Axes3D(fig1)  # 用这个绘图对象创建一个Axes对象(有3D坐标)
    X, Y = np.mgrid[-10:10:100j, -10:10:100j]  # 从-2到2分别生成40个取样坐标，并作满射联合
    Z = fun(X, Y)  # 用取样点横纵坐标去求取样点Z坐标
    plt.title("This is main title")  # 总标题
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, alpha=0.5)  # 用取样点(x,y,z)去构建曲面
    ax.set_xlabel('x label', color='r')
    ax.set_ylabel('y label', color='g')
    ax.set_zlabel('z label', color='b')  # 给三个坐标轴注明
    plt.show()  # 显示模块中的所有绘图对象


def get_extreme_point():
    # 极小值
    def gradient_func(x, y):
        return [6 * x , 8 * y]

    alpha = 0.01
    x0 = [100, 100]
    f0 = fun(x0[0], x0[1])

    for i in range(10000):
        print("[%d] (%.5f, %.5f): %.5f" % (i, x0[0], x0[1], f0))
        J = gradient_func(x0[0], x0[1])
        x1 = [x0[0]-alpha *J[0] ,x0[1] - alpha * J[1]]
        f1 = fun(x1[0], x1[1])
        if abs(f1 - f0) < 1e-9:
            print("极值点：", x0, " 极小值：", f0)
            break
        x0 = x1
        f0 = f1



if __name__ == "__main__":
    show_func()

    # get_extreme_point()