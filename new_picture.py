import argparse
import datetime
import logging
import os
from typing import List, Any

import cv2 as cv
from math import floor, sqrt

import numpy as np
from matplotlib import pyplot as plt
from numpy import array
from scipy.interpolate import CubicSpline

from bf import Graph, maximize_matrix, fill_matrix_none, coordinates_to_maximized
from inferer_land import class_land
from inferer_road import class_road
from Astar import astar_path
from spline import merge_sort


def routing_calculation(origin:str, height_root:str, output_img:str, start:tuple, end:tuple):
    # Read road image
    road = class_road(".\\outputs\\test\\cfg.yaml", origin, ".\\input\\test9_result.png")
    land = class_land(".\\land\\outputs\\test\\cfg_land.yaml", origin, ".\\input\\test9_resultL.png")

    # Read land image
    image = cv.imread(origin)
    if land is None:
        logging.error("Failed to read input image file")

    road_orig = cv.cvtColor(road, cv.COLOR_BGR2GRAY)
    land_orig = cv.cvtColor(land, cv.COLOR_BGR2GRAY)

    # roi = land_orig
    #
    # roi[road_orig == 77] = 77

    # cv.imwrite(output_img, roi)

    start_tuple = (start[1], start[0])
    end_tuple = end
    width_shrinkage = 10
    height_shrinkage = 10
    end_tuples: List[tuple] = list()
    end_tuples.append(end_tuple)
    matrix = block2matrix(land_orig, height_root)
    matrix_max = maximize_matrix(width_shrinkage, height_shrinkage, matrix)
    start_maximized = coordinates_to_maximized(start_tuple, width_shrinkage, height_shrinkage)
    finish_maximized = coordinates_to_maximized(end_tuple, width_shrinkage, height_shrinkage)
    graph = Graph(matrix_max)
    print(datetime.datetime.now())
    result = graph.find_path(start_maximized, finish_maximized)
    # print(result)
    sorted_list = result.copy()
    # what diraction
    dx = abs(end[0]-start[0])
    dy = abs(end[1] - start[1])
    position = False # True - вертекальное положение дороги,False - горизонтальное
    # horizontal
    if dx > dy:
        merge_sort(sorted_list, 0, len(sorted_list))
    # vertical
    else:
        sorted_list = list()
        position = True
        for point in result:
            sorted_list.append((point[1], point[0]))
        merge_sort(sorted_list, 0, len(sorted_list))
    # axis x in increase
    sorted_copy = sorted_list.copy()
    for i in range(len(sorted_list)-1):
        if sorted_list[i][0] == sorted_list[i+1][0]:
            sorted_copy.remove(sorted_list[i+1])
    sorted_list = sorted_copy.copy()

    # interpolation
    list_x = list()
    list_y = list()
    for i in sorted_list:
        list_x.append(i[0])
        list_y.append(i[1])
    new_coord_x = list()
    new_coord_y = list()
    step = 15
    for i in range(0, len(list_x), step):
        new_coord_x.append(list_x[i])
        new_coord_y.append(list_y[i])
    if new_coord_x[-1] != list_x[-1] or new_coord_y[-1] != list_y[-1]:
        new_coord_x.append(list_x[-1])
        new_coord_y.append(list_y[-1])
    f = CubicSpline(new_coord_x, new_coord_y, bc_type='natural')
    new_new_y = []
    start_x = list_x[0]
    end_x = list_x[-1]
    full_list_x = list()
    while start_x != end_x:
        full_list_x.append(start_x)
        start_x += 1
    full_list_x.append(end_x)
    # print(full_list_x)
    for i in range(len(full_list_x)):
        new_new_y.append(int(f(full_list_x[i])))
    final_result = list()
    for i in range(len(full_list_x)):
        if position:
            final_result.append((new_new_y[i], full_list_x[i]))
        else:
            final_result.append((full_list_x[i], new_new_y[i]))
    print(final_result)
    print(datetime.datetime.now())
    final_result_copy = final_result.copy()
    for i in range(len(final_result)-1):
        if final_result[i][1] - final_result[i + 1][1] > 1:
            const_x = final_result[i+1][0]
            const_y = final_result[i+1][1]
            for j in range(final_result[i][1] - final_result[i+1][1] - 1):
                const_y -=1
                final_result_copy.append((const_x, const_y))

    print(final_result_copy)

    new_mat = fill_matrix_none(width_shrinkage, height_shrinkage, matrix, final_result_copy)
    matrix = [[0 for _ in range(len(new_mat[0]))] for _ in range(len(new_mat))]
    matrix_ = [[0 for _ in range(len(matrix_max[0]))] for _ in range(len(matrix_max))]
    for i in range(len(new_mat)):
        for j in range(len(new_mat[0])):
            if new_mat[i][j] != None:
                matrix[i][j] = 100
    for i in range(len(final_result_copy)):
        x, y = final_result_copy[i]
        matrix_[x][y] = 1
    plt.title("Path")
    plt.imshow(matrix_)
    plt.savefig("path.png")
    mask = array(matrix)
    color_mask = np.zeros_like(image, dtype=np.uint8)
    color_mask[np.squeeze(mask == 100), :] = (0, 0, 255)
    #result = cv.addWeighted(land, 0.5, color_mask, 0.5, 1.0)
    result_img = cv.addWeighted(image, 0.5, color_mask, 0.5, 1.0)
    print("Saving result to '{0}'...".format(output_img))
    cv.imwrite(output_img, result_img)

def block2matrix(image_land, image_height):
    im_height = cv.imread(image_height)
    if im_height is None:
        logging.error("Failed to read input image file")

    # Classify land
    image_land2 = image_land
    image_land2[image_land2 == 179] = 4
    image_land2[image_land2 == 226] = 2
    image_land2[image_land2 == 106] = 1
    image_land2[image_land2 == 150] = 3
    image_land2[image_land2 == 30] = 5
    image_land2[image_land2 == 255] = 6
    image_land2[image_land2 == 0] = 7

    # Classify height
    land_height = cv.cvtColor(im_height, cv.COLOR_BGR2GRAY)
    land_height[0 <= land_height.all() <= 27] = 1
    land_height[28 <= land_height.all() <= 55] = 2
    land_height[56 <= land_height.all() <= 84] = 3
    land_height[85 <= land_height.all() <= 112] = 4
    land_height[113 <= land_height.all() <= 140] = 5
    land_height[141 <= land_height.all() <= 169] = 6
    land_height[170 <= land_height.all() <= 197] = 7
    land_height[198 <= land_height.all() <= 225] = 8
    land_height[226 <= land_height.all() <= 254] = 9
    land_height[land_height == 255] = 10
    matrix_height = array(land_height)
    matrix_land = array(image_land2)
    matrix = matrix_land * matrix_height
    matrix_list = matrix.tolist()
    return matrix_list

if __name__ == '__main__':
    routing_calculation(".\\input\\test2.png", ".\\input\\newDem.png", ".\\input\\RESULT.png", (625, 550), (1710, 1219))
    # exit(main())