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

from bf import Graph, maximize_matrix, fill_matrix_none, coordinates_to_maximized
from inferer_land import class_land
from inferer_road import class_road
from Astar import astar_path


def generated_sum_pic(origin:str, height_root:str, output_img:str, start:tuple, end:tuple):
    # Read road image
    road = class_road(".\\outputs\\test\\cfg.yaml", origin, ".\\input\\test9_result.png")
    land = class_land(".\\land\\outputs\\test\\cfg_land.yaml", origin, ".\\input\\test9_resultL.png")

    # Read land image
    image = cv.imread(origin)
    if land is None:
        logging.error("Failed to read input image file")

    # road_orig = cv.cvtColor(road, cv.COLOR_BGR2GRAY)
    land_orig = cv.cvtColor(land, cv.COLOR_BGR2GRAY)

    # roi = land_orig

    # roi[road_orig == 77] = 77

    # cv.imwrite(output_img, roi)

    # Coordinates
    # height, width = land_orig.shape
    # deg_pre_pixel_long = abs((rd[0] - lu[0])/width)
    # deg_pre_pixel_lat = abs((lu[1] - rd[1])/height)
    start_tuple = (start[1], start[0])
    end_tuple = end
    width_shrinkage = 10
    height_shrinkage = 10
    # start_tuple = int((lu[0] - start[0])//deg_pre_pixel_long), int((rd[1] - start[1])//deg_pre_pixel_lat)
    # end_tuple = int((lu[0]-end[0])//deg_pre_pixel_long), int((rd[1] - end[1])//deg_pre_pixel_lat)
    end_tuples: List[tuple] = list()
    end_tuples.append(end_tuple)
    matrix = block2matrix(land_orig, height_root)
    matrix_max = maximize_matrix(width_shrinkage, height_shrinkage, matrix)
    start_maximized = coordinates_to_maximized(start_tuple, width_shrinkage, height_shrinkage)
    finish_maximized = coordinates_to_maximized(end_tuple, width_shrinkage, height_shrinkage)
    graph = Graph(matrix_max)
    print(datetime.datetime.now())
    result = graph.find_path(start_maximized, finish_maximized)
    print(result)
    print(datetime.datetime.now())
    new_mat = fill_matrix_none(width_shrinkage, height_shrinkage, matrix, result)
    matrix = [[0 for _ in range(len(new_mat[0]))] for _ in range(len(new_mat))]
    matrix_ = [[0 for _ in range(len(matrix_max[0]))] for _ in range(len(matrix_max))]
    for i in range(len(new_mat)):
        for j in range(len(new_mat[0])):
            if new_mat[i][j] != None:
                matrix[i][j] = 100
    for i in range(len(result)):
        x, y = result[i]
        matrix_[x][y] = 1
    plt.title("Path")
    plt.imshow(matrix_)
    plt.savefig("path.png")
    mask = array(matrix)
    color_mask = np.zeros_like(image, dtype=np.uint8)
    color_mask[np.squeeze(mask == 100), :] = (0, 0, 255)
    #result = cv.addWeighted(land, 0.5, color_mask, 0.5, 1.0)
    result = cv.addWeighted(image, 0.5, color_mask, 0.5, 1.0)
    print("Saving result to '{0}'...".format(output_img))
    cv.imwrite(output_img, result)

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

# def main() -> int:
#     # Create argument parser
#     parser = argparse.ArgumentParser(description='Sum images')
#     parser.add_argument('--shot', dest='image', required=True, type=str, default="",
#                         help="Path to dataset root directory")
#     parser.add_argument('--height', dest='height', required=True, type=str, default="",
#                         help="Path to dataset root directory")
#     parser.add_argument('--output-img', dest='output_img', required=True, type=str, default="", metavar="FILE",
#                         help="path to output image")
#     parser.add_argument('--left_up', dest='left_up_point', required=True, type=tuple, default="", metavar="",
#                         help="path to output image")
#     parser.add_argument('--right_down', dest='right_down_point', required=True, type=tuple, default="", metavar="",
#                         help="path to output image")
#     parser.add_argument('--start', dest='start_point', required=True, type=tuple, default="", metavar="",
#                         help="path to output image")
#     parser.add_argument('--end', dest='end_point', required=True, type=tuple, default="", metavar="",
#                         help="path to output image")
#     args = parser.parse_args()
#
#     # generated_sum_pic(args.road_root, args.land_root, args.output_img)
#     # generated_sum_pic(".\\input\\test9_orig.png",".\\input\\test6_r.png", ".\\input\\test9_l.png", ".\\input\\test6_result.png", (57.195948, 27.880420), (57.191277, 27.899967), (0,0), (0,0))
#     print('Done.')
#     return 0

if __name__ == '__main__':
    generated_sum_pic(".\\input\\test3.png",".\\input\\newDem.png", ".\\input\\RESULT.png",  (628,296), (2422,552))
    # exit(main())