import numpy as np

"""
二分查找

"""



def binary_search(data, item):
    low = min(data)
    high = max(data)

    num = 0
    while low <= high:
        num += 1
        guess = int((low + high) / 2)
        if guess == item:
            return guess, num
        elif guess > item:
            high = guess - 1
        elif guess < item:
            low = guess + 1
    return None, num



if __name__ == "__main__":
    data = np.arange(1, 101, 1)
    item = 67
    res, num = binary_search(data, item)
    print(res, num)

    num = np.log2(4 * 1e9)
    print(np.ceil(num))