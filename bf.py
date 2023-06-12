import math

from matplotlib import pyplot as plt
import random
import datetime

from numpy import array


def maximize_matrix(new_width: int, new_height: int, matrix: list):
    divided_matrix = []
    for index in range(len(matrix)):
        row = matrix[index]
        combined_row = []
        for j in range(math.ceil(len(row) / new_width)):
            sliced = row[j * new_width:(j + 1) * new_width]
            combined_row.append(sum(sliced) / len(sliced))
        divided_matrix.append(combined_row)
    final_matrix = []
    summary = [0 for _ in range(len(divided_matrix[0]))]
    inner_counter = 0
    for index in range(0, len(divided_matrix)):
        if inner_counter < new_height:
            for inner_index in range(len(divided_matrix[index])):
                summary[inner_index] += divided_matrix[index][inner_index]
            inner_counter += 1
        if inner_counter == new_height:
            final_matrix.append(summary)
            summary = [0 for _ in range(len(divided_matrix[0]))]
            inner_counter = 0
        if (index == len(divided_matrix) - 1) and len(matrix) % new_height != 0:
            final_matrix.append(summary)
    for index in range(len(final_matrix)):
        for inner_index in range(len(final_matrix[index])):
            final_matrix[index][inner_index] = final_matrix[index][inner_index] / new_height
    return array(final_matrix)


def fill_matrix_none(width_prop, height_prop, matrix: list, coordinates: list):
    width = len(matrix[0])
    height = len(matrix)
    b = [[None for _ in range(width)] for _ in range(height)]
    for element in coordinates:
        x = element[1]
        y = element[0]
        left_top_corner_x = x * width_prop
        left_top_corner_y = y * height_prop
        right_bot_corner_x = (x + 1) * width_prop
        right_bot_corner_y = (y + 1) * height_prop
        if right_bot_corner_x > width:
            right_bot_corner_x = width
        if right_bot_corner_y > height:
            right_bot_corner_y = height
        for j in range(left_top_corner_y, right_bot_corner_y):
            for i in range(left_top_corner_x, right_bot_corner_x):
                b[j][i] = matrix[j][i]
    return b


def coordinates_to_maximized(coordinates, width_shrinkage, height_shrinkage):
    return math.floor(float(coordinates[0]) / width_shrinkage), math.floor(float(coordinates[1]) / height_shrinkage)
