#!/usr/bin/python
# -*- coding: utf-8 -*-

#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def decisions_matrix(n, q, Y_min, Y_max, k, z, g, y_0, y_last):
    decisions = [[0] * (2 * n) for _ in range(Y_max + 1)]

    for iter, col in zip(range(n - 1, -1, -1), range(0, 2 * n + 1, 2)):
        for y in range(Y_min, Y_max + 1):
            if iter == 0 and y != y_0:
                continue

            if iter == n - 1:
                if 0 <= y_last + q[iter] - y <= z:
                    x = y_last + q[iter] - y
                    f = g[x] + k[y_last]
                else:
                    x = np.inf
                    f = np.inf
                decisions[y][col], decisions[y][col + 1] = x, f
            else:
                if Y_min + q[iter] - y > z:
                    x = np.inf
                    f = np.inf

                    decisions[y][col], decisions[y][col + 1] = x, f

                else:
                    if Y_max + q[iter] - y >= z:
                        x_temp = [i for i in range(Y_min + q[iter] - y, z + 1)]
                    else:
                        x_temp = [i for i in range(Y_min + q[iter] - y, Y_max + q[iter] - y + 1)]

                    xf = []
                    for x_elem in x_temp:
                        xf.append((x_elem, g[x_elem] + k[y + x_elem - q[iter]] + decisions[y + x_elem - q[iter]][col - 1]))

                    x = []
                    min_xf = min(xf, key=lambda i: i[1])

                    if min_xf[1] == np.inf:
                        x = np.inf
                    else:
                        for xf_elem in xf:
                            if xf_elem[1] == min_xf[1]:
                                x.append(xf_elem[0])

                        if len(x) == 1:
                            x = x[0]

                    decisions[y][col], decisions[y][col + 1] = x, min_xf[1]

    return decisions


def objective_function(n, q, Y_min, Y_max, k, z, g, y_0, y_last):
    decisions = decisions_matrix(n, q, Y_min, Y_max, k, z, g, y_0, y_last)
    objective_tab = [[]]

    def _objective_function(row, col, objective_tab, decisions, object, branch, q):
        if col < 0:
            return
        else:
            if type(decisions[row][col]) is list:
                for decision in range(len(decisions[row][col])):
                    if decision != 0:
                        if len(objective_tab[0]) != 0:
                            objective_tab.append(objective_tab[branch].copy())
                        else:
                            objective_tab.append([])

                for decision in range(len(decisions[row][col])):
                    root_branch = branch
                    root_object = object
                    y = row + decisions[row][col][decision] - q[root_object]
                    objective_tab[root_branch + decision].append(decisions[row][col][decision])

                    _objective_function(y, col - 2, objective_tab, decisions, root_object + 1, root_branch + decision, q)

            else:
                y = row + decisions[row][col] - q[object]
                objective_tab[branch].append(decisions[row][col])

                _objective_function(y, col - 2, objective_tab, decisions, object + 1, branch, q)

    _objective_function(y_0, 2 * n - 2, objective_tab, decisions, 0, 0, q)

    return objective_tab


if __name__ == '__main__':
    # n = 4
    # q = [4, 2, 6, 5, np.inf, np.inf]
    # # q = [3, 3, 3, 3]
    # Y_min = 2
    # Y_max = 5
    # k = [np.inf, np.inf, 1, 2, 2, 4]
    # z = 5
    # g = [2, 8, 12, 15, 17, 20]
    # y_0 = 4
    # y_last = 3

    n = 6
    q = [4, 2, 6, 5, 3, 3]
    Y_min = 2
    Y_max = 5
    k = [np.inf, np.inf, 1, 2, 2, 4]
    z = 5
    g = [2, 8, 12, 15, 17, 20]
    y_0 = 4
    y_last = 3

    print('Objective function: ', objective_function(n, q, Y_min, Y_max, k, z, g, y_0, y_last))

    print('\nDecisions matrix:')

    dec_matrix = decisions_matrix(n, q, Y_min, Y_max, k, z, g, y_0, y_last)

    pd.DataFrame(dec_matrix).to_csv("decisions_matrix.csv")

    df = pd.read_csv('decisions_matrix.csv')
    print(df)
