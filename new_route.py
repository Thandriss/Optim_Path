import argparse
import datetime
import logging
import os
from math import sqrt
from typing import List, Any
import cv2 as cv
import numpy as np
from numpy import array
from scipy.interpolate import CubicSpline

from bf import Graph, maximize_matrix, fill_matrix_none, coordinates_to_maximized
from inferer_land import class_land
from inferer_road import class_road
from spline import merge_sort


def routing_calculation(origin: str, height_root: str, output_img: str, start: tuple, end: list):
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
    width_shrinkage = 10
    height_shrinkage = 10
    matrix = block2matrix(land_orig, height_root)
    matrix_max = maximize_matrix(width_shrinkage, height_shrinkage, matrix)
    start_maximized = coordinates_to_maximized(start_tuple, width_shrinkage, height_shrinkage)
    result = []
    list_position = []
    list_finish = []
    list_start = []
    for i in range(len(end)):
        start_early = False
        if i == 0:
            start_p = start_maximized
            end_p = coordinates_to_maximized(end[i], width_shrinkage, height_shrinkage)
            list_start.append(start_p)
            list_finish.append(end_p)
        else:
            start_p = coordinates_to_maximized(end[i - 1], width_shrinkage, height_shrinkage)
            print(start_p)
            start_p = start_p[1], start_p[0]
            end_p = coordinates_to_maximized(end[i], width_shrinkage, height_shrinkage)
            print(end_p)
            list_start.append(start_p)
            list_finish.append(end_p)
        # what diraction
        dx = abs(end_p[0] - start_p[0])
        dy = abs(end_p[1] - start_p[1])
        position = False  # True - вертекальное положение дороги,False - горизонтальное
        if dy >= dx:
            position = True
        # cutting
        if position:
            print("vertical")
            reduced_mat = matrix_max[start_p[0]:end_p[1] + 1].copy()
            # change coords
            new_start = (0, start_p[1])
            new_finish = (end_p[0], end_p[1] - start_p[0])
            print("new finish")
            print(new_finish)
            print("new start")
            print(new_start)
        else:
            print("horizontal")
            if start_p[1] > end_p[0]:
                start_early = True
                reduced_mat = matrix_max[:, end_p[0]:start_p[1] + 1].copy()
            else:
                reduced_mat = matrix_max[:, start_p[1]:end_p[0] + 1].copy()
            if start_early:
                new_start = (start_p[0], len(reduced_mat[0]) - 1)
                new_finish = (0, end_p[1])
            else:
                new_start = (start_p[0], 0)
                new_finish = (abs(end_p[0] - start_p[1]), end_p[1])
            print("new finish")
            print(new_finish)
            print("new start")
            print(new_start)
        graph = Graph(reduced_mat)
        print(datetime.datetime.now())
        path = graph.find_path(new_start, new_finish)
        print("path")
        print(path)
        print("end path")
        # processing results
        if position:
            for j in range(len(path)):
                path[j] = path[j][0] + start_p[0], path[j][1]
        else:
            for j in range(len(path)):
                if start_early:
                    path[j] = path[j][0], path[j][1] + end_p[0]
                else:
                    path[j] = path[j][0], path[j][1] + start_p[1]
        print("path")
        print(path)
        print("end path")
        result.append(path)
        list_position.append(position)

    print(result)
    sorted_lists = result
    final = []
    print("POSITION")
    print(list_position)
    print("FINISH")
    print(list_finish)
    print("Start")
    print(list_start)
    for i in range(len(sorted_lists)):
        print("NEW!!!________________________________")
        sorted_list = sorted_lists[i]
        print("sorted_list")
        print(sorted_list)
        print("end sorted_list")
        local_position = list_position[i]
        # horizontal
        if not local_position:
            merge_sort(sorted_list, 0, len(sorted_list))
        # vertical
        else:
            merge_sort(sorted_list, 0, len(sorted_list))
            sorted_list_new = list()
            for point in sorted_list:
                sorted_list_new.append((point[1], point[0]))
            # merge_sort(sorted_list_new, 0, len(sorted_list))
            sorted_list = sorted_list_new.copy()
        # axis x in increase
        sorted_copy = sorted_list.copy()
        # second index - param
        for j in range(len(sorted_list) - 1):
            if sorted_list[j][1] == sorted_list[j + 1][1]:
                sorted_copy.remove(sorted_list[j + 1])
        sorted_list = sorted_copy.copy()
        print("sorted_list")
        print(sorted_list)
        print("end sorted_list")
        # if local_position:
        #     sorted_list.remove(sorted_list[-1])
        #     sorted_list.append((list_finish[i][0], list_finish[i][1]))
        # interpolation
        list_x = list()
        list_y = list()
        print("changed")
        ind_x = 1
        ind_y = 0
        # if not local_position:
        #     ind_x = 1
        #     ind_y = 0
        for point in sorted_list:
            list_x.append(point[ind_x])
            list_y.append(point[ind_y])
        print("list x and y")
        print(list_x)
        print(list_y)
        print("end list x and y")
        new_coord_x = list()
        new_coord_y = list()
        sort_coords = list()
        step = int(sqrt(len(sorted_list))) + 11
        for ind in range(0, len(list_x), step):
            new_coord_x.append(list_x[ind])
            new_coord_y.append(list_y[ind])
            sort_coords.append((list_x[ind], list_y[ind]))
        if new_coord_x[-1] != list_x[-1] or new_coord_y[-1] != list_y[-1]:
            new_coord_x.append(list_x[-1])
            new_coord_y.append(list_y[-1])
            sort_coords.append((list_x[-1], list_y[-1]))
        new_coord_x = list()
        new_coord_y = list()
        merge_sort(sort_coords, 0, len(sort_coords))
        copy_sorted = sort_coords.copy()
        for j in range(len(sort_coords) - 1):
            if sort_coords[j][0] == sort_coords[j + 1][0]:
                copy_sorted.remove(sort_coords[j + 1])
        sort_coords = copy_sorted
        del copy_sorted
        for point in sort_coords:
            new_coord_x.append(point[0])
            new_coord_y.append(point[1])
        if local_position:  # was it
            indx = 1
            indy = 0
        else:
            indx = 0
            indy = 1
        if new_coord_x[-1] != list_finish[i][indx] and new_coord_x[0] != list_finish[i][indx]:
            # and new_coord_x[0] != list_finish[i][indx]
            print("add")
            new_coord_x.append(list_finish[i][indx])
            new_coord_y.append(list_finish[i][indy])
            list_x.append(list_finish[i][indx])
            list_y.append(list_finish[i][indy])
        elif new_coord_x[-1] == list_finish[i][indx]:
            print("here")
            new_coord_x.pop(-1)
            new_coord_y.pop(-1)
            list_x.pop(-1)
            list_y.pop(-1)
            new_coord_x.append(list_finish[i][indx])
            new_coord_y.append(list_finish[i][indy])
            list_x.append(list_finish[i][indx])
            list_y.append(list_finish[i][indy])
        if new_coord_x[0] == (list_finish[i][indx]):
            list_x.pop(0)
            list_y.pop(0)
            new_coord_x.pop(0)
            new_coord_y.pop(0)
            new_coord_x.append(list_finish[i][indx])
            new_coord_y.append(list_finish[i][indy])
        # elif new_coord_x[0] != list_finish[i][indx]:
        #     list_x.pop(0)
        #     list_y.pop(0)
        #     new_coord_x.pop(0)
        #     new_coord_y.pop(0)
        #     new_coord_x.append(list_finish[i][indx])
        #     new_coord_y.append(list_finish[i][indy])

        print("new coord")
        print(new_coord_x)
        print(new_coord_y)
        print("end new coord")
        make_coord = list()
        for j in range(len(new_coord_x)):
            make_coord.append((new_coord_x[j], new_coord_y[j]))
        merge_sort(make_coord, 0, len(make_coord))
        new_coord_x = list()
        new_coord_y = list()
        print("make coord")
        print(make_coord)
        for j in range(len(make_coord)):
            new_coord_x.append(make_coord[j][0])
            new_coord_y.append(make_coord[j][1])
        del make_coord
        print("new coord")
        print(new_coord_x)
        print(new_coord_y)
        print("end new coord")
        f = CubicSpline(new_coord_x, new_coord_y, bc_type='natural')
        new_new_y = []
        start_x = new_coord_x[0]
        end_x = new_coord_x[-1]
        full_list_x = list()
        del list_x
        del list_y
        del new_coord_y
        del new_coord_x
        print("ADD x")
        print(start_x)
        print(end_x)
        while start_x != end_x:
            full_list_x.append(start_x)
            start_x += 1
        full_list_x.append(end_x)
        for id in range(len(full_list_x)):
            new_new_y.append(int(f(full_list_x[id])))
        final_result = list()
        for id in range(len(full_list_x)):
            if local_position:
                final_result.append((full_list_x[id], new_new_y[id]))
            else:
                final_result.append((new_new_y[id], full_list_x[id]))

        print(final_result)
        print(datetime.datetime.now())
        final_result_copy = final_result.copy()
        ind_x = 0
        ind_y = 1
        for index in range(len(final_result) - 1):
            if abs(final_result[index][ind_y] - final_result[index + 1][ind_y]) > 1:
                const_x = final_result[index + 1][ind_x]
                const_y = final_result[index + 1][ind_y]
                for j in range(abs(final_result[index][ind_y] - final_result[index + 1][ind_y]) - 1):
                    const_y -= 1
                    if not local_position:
                        final_result_copy.append((const_y, const_x))
                    else:
                        final_result_copy.append((const_x, const_y))
            if abs(final_result[index][ind_x] - final_result[index + 1][ind_x]) > 1:
                if final_result[index][ind_x] < final_result[index + 1][ind_x]:
                    const_x = final_result[index + 1][ind_x]
                    const_y = final_result[index + 1][ind_y]
                else:
                    const_x = final_result[index][ind_x]
                    const_y = final_result[index][ind_y]
                for j in range(abs(final_result[index][ind_x] - final_result[index + 1][ind_x]) - 1):
                    const_x -= 1
                    if local_position:
                        final_result_copy.append((const_y, const_x))
                    else:
                        final_result_copy.append((const_x, const_y))
        final.append(final_result_copy)
        print(final_result_copy)
    print(final)
    final_path = []
    for i in range(len(final)):
        for point in final[i]:
            final_path.append(point)
    new_mat = fill_matrix_none(width_shrinkage, height_shrinkage, matrix, final_path)
    matrix = [[0 for _ in range(len(new_mat[0]))] for _ in range(len(new_mat))]
    for i in range(len(new_mat)):
        for j in range(len(new_mat[0])):
            if new_mat[i][j] != None:
                matrix[i][j] = 100
    mask = array(matrix)
    color_mask = np.zeros_like(image, dtype=np.uint8)
    color_mask[np.squeeze(mask == 100), :] = (0, 0, 255)
    result_img = cv.addWeighted(land, 0, color_mask, 1, 1.0)
    result_ph = cv.addWeighted(image, 0.5, color_mask, 0.5, 1.0)
    print("Saving result to '{0}'...".format(output_img))
    cv.imwrite(output_img, result_img)
    cv.imwrite(".\\input\\RESULT_PH.png", result_ph)


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
    matrix_list = matrix.copy()
    return matrix_list


if __name__ == '__main__':
    routing_calculation(".\\input\\test4.png", ".\\input\\test4_h.png", ".\\input\\RESULT.png", (2056, 15),
                        [(1101, 925), (718, 941), (172, 1413)])

    # exit(main())
