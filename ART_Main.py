# -*- coding: utf-8 -*-

import numpy as np
from ART_classify import ART_classify

#przykladowy obraz wejsciowy
image = np.array([[0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 1, 1, 1, 1, 1, 1]])

#wyjscia sieci i przyporzadkowane klasy
y = np.zeros([10])
y_number = 10
classes = np.zeros([10])
classes[0] = 1

#inicjalizacja wag
n = image.size 
w = np.empty([n, y_number])
v = np.empty([n, y_number])

for j in range(0, y_number):
    for i in range(0, n):
        v[i][j] = 1
        w[i][j] = 1/(1+n)

# wprowadzenie progu czujnosci
prog_czujnosci = 0.999

v, w, classes = ART_classify(image, y, y_number, classes, n, w, v, prog_czujnosci)
