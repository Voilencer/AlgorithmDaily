import numpy as np
from scipy.spatial.distance import pdist



def euclidean_distance(A, B):
    # pdist(np.vstack([A,B]))
    return np.sqrt(np.sum(np.square(np.array(A) - np.array(B))))


def manhattan_distance(A, B):
    # pdist(np.vstack([A,B]), 'cityblock')
    return np.sum(np.abs(np.array(A) - np.array(B)))

def chebyshev_distance(A, B):
    # pdist(np.vstack([A,B]), 'chebyshev')
    return np.max(np.abs(np.array(A) - np.array(B)))

# p=1 曼哈顿; p=2 欧氏距离; p->00 切比雪夫距离
def minkowski_distance(A, B):
    X = np.vstack([A,B])
    return pdist(X, "minkowski", p = 2)

def standardized_euclidean_distance(A, B):
    X = np.vstack([A,B])
    sk = np.var(X, axis = 0, ddof = 1)
    return np.sqrt(((np.array(A) - np.array(B)) **2 / sk).sum())

def hamming_distance(A, B):
    num = 0
    for index in range(len(A)):
        if A[index] != B[index]:
            num += 1
    return num

def cosine(A, B):
    d1 = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
    # d2 = 1 - pdist(np.vstack([A,B]), 'cosine')
    return d1

def X2_distance(x1, x2):
    tmp = []
    if len(x1) != len(x2):
        return
    for i in range(len(x1)):
        dist0 = (x1[i] - x2[i])**2 / (x1[i] + x2[i] + 0.00001)
        tmp.append(dist0)
    dist = sum(tmp)
    return dist



if __name__ == "__main__":
    A = (1,1)
    B = (2,3)

    dist = euler_dist(A, B)
    print("euler dist = ", dist)