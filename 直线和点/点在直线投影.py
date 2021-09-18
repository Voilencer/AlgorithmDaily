import math
import matplotlib.pyplot as plt

"""
直线：P1P2
直线外一点：P3
（1）




"""


def cal_proj1(P1, P2, P3):
    # 直线求交点
    if P1[0] == P2[0]:
        x = P1[0]
        y = P3[1]
    elif P1[1] == P2[1]:
        x = P3[0]
        y = P1[1]
    else:
        k_12 = (P2[1]-P1[1]) / (P2[0] - P1[0])
        b_12 = P1[1] - k_12 * P1[0]

        k_03 = -1/k_12
        b_03 = P3[1] - k_03 * P3[0]
        x = (b_03 - b_12) / (k_12 - k_03)
        y = k_12 * x + b_12
    return [x, y]


def cal_proj2(P1, P2, P3):
    # 向量解法
    P_12 = [P2[0]-P1[0], P2[1]-P1[1]]
    P_13 = [P3[0]-P1[0], P3[1]-P1[1]]

    k = (P_12[0] * P_13[0] + P_12[1] * P_13[1]) / (math.pow(P_12[0], 2) + math.pow(P_12[1], 2))
    x = k * P_12[0] + P1[0]
    y = k * P_12[1] + P1[1]
    return [x, y]


def show_res():
    figure = plt.figure()
    plt.scatter([P1[0], P2[0], P3[0]], [P1[1], P2[1], P3[1]])
    plt.plot([P1[0], P2[0]], [P1[1], P2[1]])

    plt.scatter(P0[0], P0[1], color='red')

    plt.scatter(0, 0, color='red')
    plt.show()



if __name__ == "__main__":

    P1 = [392, 452]
    P2 = [299, 274]

    P3 = [253, 165]
    P0 = cal_proj1(P1, P2, P3)
    print(P0)

    # P0 = cal_proj2(P1, P2, P3)
    # print(P0)

    # P12 = [P1[0]-P2[0], P1[1]-P2[1]]
    # P03 = [P3[0]-P0[0], P3[1]-P0[1]]
    # res = P12[0] * P03[0] + P12[1] * P03[1]
    # print(res)

    show_res()