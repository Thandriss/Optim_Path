import unittest
from math import sqrt

from dijkstra import find_path


class MyTestCase(unittest.TestCase):
    def test_something(self):
        matrix1 = [[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]]
        start = (0, 0)
        ends = [(2, 2)]
        result = find_path(start, ends, matrix1)
        print(result)
        path = result[0][0]
        print(path)
        self.assertEqual(result[0][1][len(result[0][1]) - 1], sqrt(2) * 2)

        matrix2 = [[1, 1, 1],
                   [1, 100, 1],
                   [1, 1, 1]]
        start = (0, 0)
        ends = [(2, 2)]
        result2 = find_path(start, ends, matrix2)
        print(result2)
        self.assertEqual(result2[0][1][len(result2[0][1]) - 1], sqrt(2) + 2)
        matrix3 = [[1, 1, 1, 4],
                   [1, 100, 10, 5],
                   [1, 1, 1, 1]]
        start = (0, 0)
        ends = [(2, 2)]
        result2 = find_path(start, ends, matrix3)
        print(result2)
        self.assertEqual(result2[0][1][len(result2[0][1]) - 1], sqrt(2) + 2)


if __name__ == '__main__':
    unittest.main()
