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


def show_image(wnd_name, img, delay = 0):
    cv2.namedWindow(wnd_name, cv2.WINDOW_NORMAL)
    cv2.imshow(wnd_name, img)
    cv2.waitKey(delay)



def myNMS(bboxes, overlap_thr = 0.5):
    keep = []
    if len(bboxes) == 0:
        return keep
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    w = bboxes[:,2]
    h = bboxes[:,3]
    x2 = bboxes[:,0] + bboxes[:,2] - 1
    y2 = bboxes[:,1] + bboxes[:,3] - 1

    areas = w * h
    scores = 1 - abs(areas - AREA0) / AREA0 # 根据实际场景，可以是置信度，或者自定义函数计算score
    order = scores.argsort()[::-1]
    while len(order) > 0:
        idx = order[0]
        keep.append(bboxes[idx])
        xx1 = np.maximum(x1[idx], x1[order[1:]])
        xx2 = np.minimum(x2[idx], x2[order[1:]])
        yy1 = np.maximum(y1[idx], y1[order[1:]])
        yy2 = np.minimum(y2[idx], y2[order[1:]])
        ww = np.maximum(0, xx2 - xx1 + 1)
        hh = np.maximum(0, yy2 - yy1 + 1)
        inter_area = ww * hh
        iou = inter_area / (areas[idx] + areas[order[1:]] - inter_area)
        inds = np.where(iou <= overlap_thr)[0]
        order = order[inds + 1]
    return keep


def mySoftNMS(bboxes, overlap_thr=0.3, Nt = 0.001, sigma=0.5):
    """
    :param bboxes:
    :param overlap_thr: IOU阈值
    :param Nt: 综合IOU和置信度的评判阈值
    :return:
    """
    keep = []
    if len(bboxes) == 0:
        return keep
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    w = bboxes[:,2]
    h = bboxes[:,3]
    x2 = bboxes[:,0] + bboxes[:,2] - 1
    y2 = bboxes[:,1] + bboxes[:,3] - 1

    areas = w * h
    scores = 1 - abs(areas - AREA0) / AREA0 # 根据实际场景，可以是置信度，或者自定义函数计算score
    order = scores.argsort()[::-1]
    while len(order) > 0:
        idx = order[0]

        keep.append(bboxes[idx])
        xx1 = np.maximum(x1[idx], x1[order[1:]])
        xx2 = np.minimum(x2[idx], x2[order[1:]])
        yy1 = np.maximum(y1[idx], y1[order[1:]])
        yy2 = np.minimum(y2[idx], y2[order[1:]])
        ww = np.maximum(0, xx2 - xx1 + 1)
        hh = np.maximum(0, yy2 - yy1 + 1)
        inter_area = ww * hh
        iou = inter_area / (areas[idx] + areas[order[1:]] - inter_area)

        S = scores[order[1:]]
        tmp = np.where(iou > overlap_thr)
        method = 0
        if method == 1: # linear
            S[tmp] = (1 - iou[tmp]) * S[tmp]
        elif method == 2: # Gaussian
            S[tmp] = np.exp(-(iou[tmp]**2)/sigma) * S[tmp]
        else:   # NMS
            S[tmp] = 0
        inds = np.where(S <= Nt)[0]
        order = order[inds + 1]
    return keep




path = r'1.jpg'
img = cv2.imread(path, -1)
AREA0 = img.shape[0] * img.shape[1] / 12.0

if __name__ == "__main__":
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create()
    regions, bboxes = mser.detectRegions(img_gray)

    imgShow = img.copy()
    for bbox in bboxes:
        cv2.rectangle(imgShow, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0,0,255), 2)
    show_image('mser', imgShow)

    # keep_bboxes = myNMS(bboxes, 0.01)
    keep_bboxes = mySoftNMS(bboxes)

    imgShow2 = img.copy()
    for bbox in keep_bboxes:
        cv2.rectangle(imgShow2, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0,0,255), 2)
    show_image('mser_nms', imgShow2)

