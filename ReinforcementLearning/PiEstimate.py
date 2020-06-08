# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 12:47:56 2020
@author: Sila
"""

# Monte Carlo methods, or Monte Carlo experiments, are a broad class of computational algorithms
# that rely on repeated random sampling to obtain numerical results.

import numpy as np
import math
import random
import matplotlib.pyplot as plt

# Initialization
square_size = 1
points_inside_circle = 0
points_inside_square = 0
sample_size = 1000
arc = np.linspace(0, np.pi / 2, 100)


# A function called generate_points which generates random points inside the square.
def generate_points(size):
    x = random.random() * size
    y = random.random() * size
    return (x, y)


# A function called is_in_circle which will check if the point we generated falls within the circle.
def is_in_circle(point, size):
    return math.sqrt(point[0] ** 2 + point[1] ** 2) <= size


# square size = r^2 = 1, circle size = r^2*pi / 4 = 1^2* pi / 4 = pi / 4.
# circle_size / square_size = pi / 4
# The program keeps track of how many points it's picked so far (N) and how many of those points fell inside the circle (M).
# Pi is then approximated as follows:
#
#
#  pi =  4*M / N
#
#
def compute_pi(points_inside_circle, points_inside_square):
    return 4 * (points_inside_circle / points_inside_square)


plt.axes().set_aspect('equal')
plt.plot(1 * np.cos(arc), 1 * np.sin(arc))

for i in range(sample_size):
    point = generate_points(square_size)
    plt.plot(point[0], point[1], 'c.')
    points_inside_square += 1

    if is_in_circle(point, square_size):
        points_inside_circle += 1

plt.show()

print("Approximate value of pi is {}".format(compute_pi(points_inside_circle, points_inside_square)))
print("Awesome")
