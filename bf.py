import math

from matplotlib import pyplot as plt
import random
import datetime

from numpy import array


class Graph:
    def __init__(self, vertices: list):
        self.width = len(vertices[0])
        self.height = len(vertices)
        self.graph = []
        for i in range(len(vertices)):
            for j in range(len(vertices[i])):
                if i > 0:
                    self.graph.append([(i, j), (i - 1, j), vertices[i - 1][j]])
                if i < len(vertices) - 1:
                    self.graph.append([(i, j), (i + 1, j), vertices[i + 1][j]])
                if j > 0:
                    self.graph.append([(i, j), (i, j - 1), vertices[i][j - 1]])
                if j < len(vertices[i]) - 1:
                    self.graph.append([(i, j), (i, j + 1), vertices[i][j + 1]])

    def bellman_ford(self, source: tuple):
        distance = [[float("Inf") for _ in range(self.width)] for _ in range(self.height)]
        distance[source[0]][source[1]] = 0
        predecessor = [[(None, None) for _ in range(self.width)] for _ in range(self.height)]
        for i in range(self.width * self.height - 1):
            print(i)
            for u, v, w in self.graph:
                if w is None:
                    continue
                if distance[u[0]][u[1]] != float("Inf") and distance[u[0]][u[1]] + w < distance[v[0]][v[1]]:
                    distance[v[0]][v[1]] = distance[u[0]][u[1]] + w
                    predecessor[v[0]][v[1]] = u
        return predecessor, distance

    def find_path(self, source: tuple, finish: tuple):
        predecessor, distance = self.bellman_ford(source)
        path = []
        current = finish[1],finish[0]
        while current != (None, None):
            path.append(current)
            current = predecessor[current[0]][current[1]]
        path.reverse()
        return path

def maximize_matrix(new_width: int, new_height: int, matrix: list):
    divided_matrix = []
    for index in range(len(matrix)):
        row = matrix[index]
        combined_row = []
        for j in range(math.ceil(len(row)/new_width)):
            sliced = row[j*new_width:(j + 1)*new_width]
            combined_row.append(sum(sliced)/len(sliced))
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
        if (index == len(divided_matrix)-1) and len(matrix) % new_height != 0:
            final_matrix.append(summary)
    for index in range(len(final_matrix)):
        for inner_index in range(len(final_matrix[index])):
            final_matrix[index][inner_index] = final_matrix[index][inner_index] / new_height
    return final_matrix

def fill_matrix_none(width_prop, height_prop, matrix:list, coordinates:list):
    width = len(matrix[0])
    height = len(matrix)
    b = [[None for _ in range(width)] for _ in range(height)]
    for element in coordinates:
        x = element[1]
        y = element[0]
        left_top_corner_x = x*width_prop
        left_top_corner_y = y*height_prop
        right_bot_corner_x = (x+1) * width_prop
        right_bot_corner_y = (y+1) * height_prop
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

if __name__ == "__main__":
    width = 10
    height = 10
    width_shrinkage = 2
    height_shrinkage = 2
    start = (0, 0)
    finish = (9, 9)
    start_maximized = coordinates_to_maximized(start, width_shrinkage, height_shrinkage)
    finish_maximized = coordinates_to_maximized(finish, width_shrinkage, height_shrinkage)
    matrix = [[random.randint(1, 100) for _ in range(width)] for _ in range(height)]
    mask = array(matrix)
    print(mask)
    print(mask[0:3])
    # print(matrix[0:1, 0:1])
    maximized = maximize_matrix(width_shrinkage, height_shrinkage, matrix)
    g = Graph(maximized)
    now = datetime.datetime.now()
    print(now)
    path = g.find_path(start_maximized, finish_maximized) #(height - 1, width - 1))
    now = datetime.datetime.now()
    print(now)
    new_mat = fill_matrix_none(width_shrinkage, height_shrinkage, matrix, path)
    g1 = Graph(new_mat)
    now = datetime.datetime.now()
    print(now)
    path2 = g1.find_path(start, finish)
    now = datetime.datetime.now()
    print(now)
    plt.figure()
    plt.title("Graph")
    plt.imshow(matrix)
    plt.savefig("mat.png")
    matrix = [[0 for _ in range(width)] for _ in range(height)]
    for i in range(len(path2)):
        x, y = path2[i]
        matrix[x][y] = 1
    plt.figure()
    plt.title("Path")
    plt.imshow(matrix)
    plt.savefig("path1.png")


