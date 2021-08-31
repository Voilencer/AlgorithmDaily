import cv2
import numpy as np



"""
findHomography(srcPoints, 
        dstPoints, 
        method=None, 
        ransacReprojThreshold=None, 
        mask=None, 
        maxIters=None, 
        confidence=None)
"""


def calib(coorA, coorB):
    # 平面A的坐标到平面B的坐标转换 s*B = H*A
    if len(coorA) != len(coorB):
        return None
    num_points = len(coorA)
    Bval, Lval = [], []
    for i in range(num_points):
        x, y = coorB[i][0], coorB[i][1]
        Xw, Yw = coorA[i][0], coorA[i][1]
        Bval.append([Xw, Yw, 1, 0, 0, 0, -x * Xw, -x * Yw])
        Bval.append([0, 0, 0, Xw, Yw, 1, -y * Xw, -y * Yw])
        Lval.extend([[x], [y]])
    Bmat = np.matrix(Bval)
    Lmat = np.matrix(Lval)
    gH = (Bmat.T * Bmat).I * (Bmat.T * Lmat)
    Hmat = np.vstack((gH, np.matrix([1])))
    Hmat = Hmat.reshape([3, 3])
    Vmat = Bmat * gH - Lmat
    if Bmat.shape[0] != Bmat.shape[1]:
        sigma = np.sqrt(Vmat.T * Vmat / (Bmat.shape[0] - Bmat.shape[1]))
    else:
        sigma = np.sqrt(Vmat.T * Vmat)
    return Hmat, np.sum(sigma)


def hom_by_cv2(coorA, coorB, method=0):
    # coorA = [x[::-1] for x in coorA]
    # coorB = [x[::-1] for x in coorB]
    # ransacReprojThreshold = None, mask = None, maxIters = None, confidence = None
    Hmat, _ = cv2.findHomography(np.array(coorA), np.array(coorB), method)
    sigma = 0
    for ind, item in enumerate(coorA):
        item_ = trans1(np.matrix(Hmat), item)
        sigma += (item_[0] -  coorB[ind][0]) ** 2 + (item_[1] -  coorB[ind][1]) ** 2
    if len(coorA) == 4:
        sigma = np.sqrt(np.sum(sigma))
    else:
        sigma = np.sqrt(np.sum(sigma) / (len(coorA) - 4))
    return np.matrix(Hmat), sigma


def trans1(Hmat, radarCoor, shape=(2160, 4096)):
    # sm = HM (已知M，求m)
    try:
        radarCoor.append(1)
        radarCoor_T = np.matrix(radarCoor).T
        tmp = Hmat * radarCoor_T
        s = np.array(tmp)[2][0]
        x = np.array(tmp)[0][0] / s
        y = np.array(tmp)[1][0] / s
        x = min(max(x, 0), shape[1])
        y = min(max(y, 0), shape[0])
    except Exception as e:
        print("trans: " + str(e))
        print('行号: ', e.__traceback__.tb_lineno)
        exit()
    return [x, y]



def trans2(Hmat, radarCoor, shape=(2160, 4096)):
    # sm = HM (已知m，求M)
    try:
        radarCoor.append(1)
        radarCoor_T = np.matrix(radarCoor).T
        tmp = Hmat.I * radarCoor_T
        s = np.array(tmp)[2][0]
        x = np.array(tmp)[0][0] / s
        y = np.array(tmp)[1][0] / s
        x = min(max(x, 0), shape[1])
        y = min(max(y, 0), shape[0])
    except Exception as e:
        print("trans: " + str(e))
        print('行号: ', e.__traceback__.tb_lineno)
        exit()
    return [x, y]


def read_label(path_label):
    with open(path_label, 'r') as fp:
        lines = fp.readlines()
    coorA, coorB = [], []
    for line in lines:
        tmp = line.replace('\n', '').split(' ')
        tmp = [float(x) for x in tmp if len(x) > 0]
        coorA.append([tmp[0], tmp[1]])
        coorB.append([tmp[2], tmp[3]])
    return coorA, coorB



if __name__ == "__main__":
    # path = r'label_2.txt'
    # coorA, coorB = read_label(path)
    coorA = [[72.0, 285.0], [452.0, 293.0], [526.0, 371.0], [556.0, 434.0], [226.0, 397.0], [82.0, 436.0], [291.0, 308.0]]
    coorB = [[617.0, 1093.0], [3296.0, 1150.0], [3799.0, 1708.0], [3974.0, 2141.0], [1646.0, 1881.0], [675.0, 2104.0], [2133.0, 1258.0]]

    Hmat, sigma = calib(coorA[:-1], coorB[:-1])
    print(Hmat, sigma)
    res = trans1(Hmat, [291, 308])
    print('A->B', res)
    res = trans2(Hmat, [2133, 1258])
    print('B->A', res)

    Hmat, sigma = hom_by_cv2(coorA[:-1], coorB[:-1])
    print(Hmat, sigma)
    res = trans1(Hmat, [291, 308])
    print('A->B', res)
    res = trans2(Hmat, [2133, 1258])
    print('B->A', res)


