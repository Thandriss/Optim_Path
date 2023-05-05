import random
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline


def merge_sort(alist, start, end):
    if end - start > 1:
        mid = (start + end) // 2
        merge_sort(alist, start, mid)
        merge_sort(alist, mid, end)
        merge_list(alist, start, mid, end)


def merge_list(alist, start, mid, end):
    left = alist[start:mid]
    right = alist[mid:end]
    k = start
    i = 0
    j = 0
    while (start + i < mid and mid + j < end):
        if (left[i][0] <= right[j][0]):
            alist[k] = left[i]
            i = i + 1
        else:
            alist[k] = right[j]
            j = j + 1
        k = k + 1
    if start + i < mid:
        while k < end:
            alist[k] = left[i]
            i = i + 1
            k = k + 1
    else:
        while k < end:
            alist[k] = right[j]
            j = j + 1
            k = k + 1


if __name__ == "__main__":
    width = 31
    coord_x = [i for i in range(width)]
    coord_y = [random.randint(0, 3) for _ in range(width)]
    new_coord_x = list()
    new_coord_y = list()
    step = 5
    for i in range(0, len(coord_y), step):
        new_coord_x.append(coord_x[i])
        new_coord_y.append(coord_y[i])

    if new_coord_x[-1] != coord_x[-1] or new_coord_y[-1] != coord_y[-1]:
        new_coord_x.append(coord_x[-1])
        new_coord_y.append(coord_y[-1])
    f = CubicSpline(new_coord_x, new_coord_y, bc_type='natural')
    new_new_y = []
    for i in range(len(coord_x)):
        new_new_y.append(int(f(coord_x[i])))
    plt.figure()
    plt.plot(coord_x, coord_y, label="a")
    plt.plot(coord_x, new_new_y, label="b")
    plt.legend()
    plt.show()
    # print(trans_y)
    # for i in range(1, len(trans_y)):
    #     new_list.append(new_list[i-1]+trans_y[i])
    # plt.figure()
    # plt.plot(list_x, list_y, label="a")
    # plt.plot(list_x, new_list, label="b")
    # plt.legend()
    # plt.show()
