import math
import cv2 as cv

from sort import merge_sort


def points(image, flag):
    read = cv.imread(image)
    road_orig = cv.cvtColor(read, cv.COLOR_BGR2GRAY)
    list_of_points = list()
    roi = road_orig
    roi[road_orig == 76] = 1
    if flag:
        for i in range(len(roi)):
            for j in range(len(roi[0])):
                if roi[i][j] == 1 and roi[i][j - 1] != 1:
                    list_of_points.append((i, j))
    else:
        for i in range(len(roi)):
            for j in range(len(roi[0])):
                if roi[i][j] == 1 and roi[i - 1][j] != 1:
                    list_of_points.append((i, j))
    return list_of_points


def deltaAngles(alpha1, alpha2):
    dAlpha = abs(alpha1 - alpha2)
    if dAlpha > math.pi:
        dAlpha = 2.0 * math.pi - dAlpha
    return dAlpha


def len_dist(p1, p2):
    px = (p1[0] - p2[0])
    py = (p1[1] - p2[1])
    return int(math.sqrt(px * px + py * py))


def angleCalc(p1, p2):
    return math.atan2((p2[1] - p1[1]), (p2[0] - p1[0]))


def compare_curve(img1, flag1, img2, flag2):
    list_orig = points(img1, flag1)
    list_draw = points(img2, flag2)
    if flag1:
        merge_sort(list_orig, 0, len(list_orig))
        merge_sort(list_draw, 0, len(list_draw))
    else:
        list_orig = sort_vert(list_orig)
        list_draw = sort_vert(list_draw)

    list1, length1 = fillPoints(list_orig, flag1)
    list2, length2 = fillPoints(list_draw, flag2)
    i = 0
    j = 0
    point1 = list1[i]
    point2 = list2[j]
    diff = 0.0
    currentDistance = min(point1.dist, point2.dist)
    prevDistance = 0.0
    while currentDistance <= 1.0:
        d = currentDistance - prevDistance
        diff += deltaAngles(point1.angle, point2.angle) * d

        if currentDistance == 1.0:
            break

        if point1.dist <= currentDistance:
            i += 1
            point1 = list1[i]

        if point2.dist <= currentDistance:
            j += 1
            point2 = list2[j]

        prevDistance = currentDistance
        currentDistance = min(point1.dist, point2.dist)

    result = list()
    result.append(1 - diff / math.pi)
    result.append(1 - abs(length1 - length2) / length1)
    result.append(length1)
    result.append(length2)
    return result


def sort_vert(list_of_points):
    new_list = list()
    for i in range(len(list_of_points)):
        new_list.append((list_of_points[i][1], list_of_points[i][0]))
    merge_sort(new_list, 0, len(new_list))
    result = list()
    for i in range(len(new_list)):
        result.append((new_list[i][1], new_list[i][0]))
    return result


def fillPoints(list_points, flag):
    new_list_points = list()
    prevPoint = tuple()
    i = 0
    if flag:
        prevPoint = list_points[i][1], list_points[i][0]
    else:
        prevPoint = list_points[i]
    length = 0
    i += 1
    while i != len(list_points) - 1:
        if flag:
            point = list_points[i][1], list_points[i][0]
        else:
            point = list_points[i]
        distance = len_dist(prevPoint, point)
        if distance < 20:
            i += 1
            continue
        length += distance
        ang = angleCalc(point, prevPoint)
        if ang < 0.0:
            ang = abs(ang)
        new_list_points.append(
            Point(
                point[0],
                point[1],
                ang,
                length
            )
        )

        prevPoint = point
        i += 1
    i = 0
    while i != len(new_list_points):
        new_list_points[i].dist /= length
        i += 1
    return new_list_points, length


class Point:

    def __init__(self, x, y, angle, len):
        self.x = x
        self.y = y
        self.angle = angle
        self.dist = len

def cost(image, land,  flag):
    list_orig = allway(image)#points(image, flag)
    read = cv.imread(land)
    land_orig = cv.cvtColor(read, cv.COLOR_BGR2GRAY)
    result_cost = 0
    list_p, _ = fillPoints(list_orig, flag)

    image_land2 = land_orig
    image_land2[image_land2 == 179] = 4
    image_land2[image_land2 == 226] = 2
    image_land2[image_land2 == 106] = 1
    image_land2[image_land2 == 105] = 1
    image_land2[image_land2 == 150] = 3
    image_land2[image_land2 == 30] = 5
    image_land2[image_land2 == 255] = 6
    image_land2[image_land2 == 0] = 7
    for i in list_p:
        if flag:
            result_cost += image_land2[i.y][i.x]
        else:
            result_cost += image_land2[i.x][i.y]
    return result_cost
def allway(img):
    read = cv.imread(img)
    road_orig = cv.cvtColor(read, cv.COLOR_BGR2GRAY)
    list_of_points = list()
    roi = road_orig
    roi[road_orig > 0] = 1
    for i in range(len(roi)):
        for j in range(len(roi[0])):
            if roi[i][j] == 1:
                list_of_points.append((i, j))
    return list_of_points

if __name__ == "__main__":
    result1 = compare_curve("D:\\proj\\Optim_Path\\test\\test1\\orig_cut.png", True,
                            "D:\\proj\\Optim_Path\\test\\test1\\RESULT_cut.png", True)
    result_costt1 = cost("D:\\proj\\Optim_Path\\test\\test1\\orig_cut.png", "D:\\proj\\Optim_Path\\test\\test1\\landshift_cut.png", True)
    result_costt2 = cost("D:\\proj\\Optim_Path\\test\\test1\\RESULT_cut.png",
                         "D:\\proj\\Optim_Path\\test\\test1\\landshift_cut.png", True)
    print("test1")
    print(result1[0], "|", result1[1], "|", result1[2], "|", result1[3], "|", result_costt1, "|", result_costt2)



    result1 = compare_curve("D:\\proj\\Optim_Path\\test\\test2\\orig_cut.png", False,
                            "D:\\proj\\Optim_Path\\test\\test2\\result_cut.png", False)
    result_costt1 = cost("D:\\proj\\Optim_Path\\test\\test2\\orig_cut.png",
                         "D:\\proj\\Optim_Path\\test\\test2\\land_cut.png", False)
    result_costt2 = cost("D:\\proj\\Optim_Path\\test\\test2\\result_cut.png",
                         "D:\\proj\\Optim_Path\\test\\test2\\land_cut.png", False)
    print("test2")
    print(result1[0], "|", result1[1], "|", result1[2], "|", result1[3], "|", result_costt1, "|", result_costt2)



    result1 = compare_curve("D:\\proj\\Optim_Path\\test\\test3\\test3_nn_cut.png", False,
                            "D:\\proj\\Optim_Path\\test\\test3\\result_cut.png", False)
    result_costt1 = cost("D:\\proj\\Optim_Path\\test\\test3\\test3_nn_cut.png",
                         "D:\\proj\\Optim_Path\\test\\test3\\landshift_cut.png", False)
    result_costt2 = cost("D:\\proj\\Optim_Path\\test\\test3\\result_cut.png",
                         "D:\\proj\\Optim_Path\\test\\test3\\landshift_cut.png", False)
    print("test3")
    print(result1[0], "|", result1[1], "|", result1[2], "|", result1[3], "|", result_costt1, "|", result_costt2)


    result1 = compare_curve("D:\\proj\\Optim_Path\\test\\test4\\orig_cut.png", False,
                            "D:\\proj\\Optim_Path\\test\\test4\\result_cut.png", False)
    result_costt1 = cost("D:\\proj\\Optim_Path\\test\\test4\\orig_cut.png",
                         "D:\\proj\\Optim_Path\\test\\test4\\land_cut.png", False)
    result_costt2 = cost("D:\\proj\\Optim_Path\\test\\test4\\result_cut.png",
                         "D:\\proj\\Optim_Path\\test\\test4\\land_cut.png", False)
    print("test4")
    print(result1[0], "|", result1[1], "|", result1[2], "|", result1[3], "|", result_costt1, "|", result_costt2)


    result1 = compare_curve("D:\\proj\\Optim_Path\\test\\test5\\orig_cut.png", False,
                            "D:\\proj\\Optim_Path\\test\\test5\\result_cut.png", False)
    result_costt1 = cost("D:\\proj\\Optim_Path\\test\\test5\\orig_cut.png",
                         "D:\\proj\\Optim_Path\\test\\test5\\land_cut.png", False)
    result_costt2 = cost("D:\\proj\\Optim_Path\\test\\test5\\result_cut.png",
                         "D:\\proj\\Optim_Path\\test\\test5\\land_cut.png", False)
    print("test5")
    print(result1[0], "|", result1[1], "|", result1[2], "|", result1[3], "|", result_costt1, "|", result_costt2)
