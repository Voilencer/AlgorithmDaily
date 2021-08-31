import cv2
import numpy as np



def IOU(bbox1, bbox2):
    l1, t1, w1, h1 = bbox1
    r1, b1 = l1 + w1, t1 + h1
    l2, t2, w2, h2 = bbox2
    r2, b2 = l2 + w2, t2 + h2
    area1 = w1 * h1
    area2 = w2 * h2
    area_inter = (min(r1, r2) - max(l1,l2)) * (min(b1,b2) - max(t1,t2))
    area_union = area1 + area2 - area_inter
    return area_inter*1.0 / area_union


def GIOU(bbox1, bbox2):
    l1, t1, w1, h1 = bbox1
    r1, b1 = l1 + w1, t1 + h1
    l2, t2, w2, h2 = bbox2
    r2, b2 = l2 + w2, t2 + h2
    area1 = w1 * h1
    area2 = w2 * h2
    area_inter = (min(r1, r2) - max(l1,l2)) * (min(b1,b2) - max(t1,t2))
    area_union = area1 + area2 - area_inter
    area_C = (max(r1, r2) - min(l1, l2)) * (max(b1, b2) - min(t1, t2))
    res_iou = area_inter*1.0 / area_union
    return res_iou - (area_C - area_union)*1.0 / area_C


def DIOU(bbox1, bbox2):
    def cal_dist(coor1, coor2):
        return np.sqrt((coor1[0]-coor2[0]) **2 + (coor1[1]-coor2[1]) ** 2)

    l1, t1, w1, h1 = bbox1
    r1, b1 = l1 + w1, t1 + h1
    l2, t2, w2, h2 = bbox2
    r2, b2 = l2 + w2, t2 + h2
    area1 = w1 * h1
    area2 = w2 * h2
    area_inter = (min(r1, r2) - max(l1,l2)) * (min(b1,b2) - max(t1,t2))
    area_union = area1 + area2 - area_inter
    area_C = (max(r1, r2) - min(l1, l2)) * (max(b1, b2) - min(t1, t2))
    res_iou = area_inter*1.0 / area_union
    rou = cal_dist([l1+w1/2.0, r1+h1/2.0], [l2+w2/2.0, r2+h2/2.0])
    c = cal_dist([min(l1,l2), min(t1,t2)], [max(r1,r2), max(b1,b2)])
    return 1 - res_iou + rou ** 2 / c ** 2


def CIOU(bbox1, bbox2, normaled=False):
    l1, t1, w1, h1 = bbox1
    r1, b1 = l1 + w1, t1 + h1
    l2, t2, w2, h2 = bbox2
    r2, b2 = l2 + w2, t2 + h2
    res_iou = IOU(bbox1, bbox2)
    res_diou = DIOU(bbox1, bbox2)
    arctan = np.atan(w2 / h2) - np.atan(w1 / h1)
    v = (4 / (np.pi ** 2)) * (np.atan(w2 / h2) - np.atan(w1 / h1))**2
    S = 1 - res_iou
    alpha = v / (S + v)
    w_temp = 2 * w1
    distance = w1 ** 2 + h1 ** 2
    ar = (8 / (np.pi ** 2)) * arctan * ((w1 - w_temp) * h1)
    if not normaled:
        cious = res_diou - (alpha * ar / distance)
    else:
        cious = res_diou - (alpha * ar)
    cious = np.clip(cious, a_min=-1.0, a_max=1.0)
    return cious





if __name__ == "__main__":
    # rect (l,t,r,b)
    # bbox x,y,w,h
    # rotbox center(), size(), angle
    bbox1 = [273.5, 215.5, 125.0, 83.0] # (x,y,w,h)
    bbox2 = [332.5, 248.5, 143.0, 99.0]
    res_iou = IOU(bbox1, bbox2)
    print("IOU = ", res_iou)

    res_giou = GIOU(bbox1, bbox2)
    print("GIOU = ", res_giou)

    res_diou = DIOU(bbox1, bbox2)
    print("DIOU = ", res_diou)

    res_ciou = DIOU(bbox1, bbox2)
    print("CIOU = ", res_ciou)


