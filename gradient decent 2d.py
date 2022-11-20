# https://moonbooks.org/Articles/How-to-implement-a-gradient-descent-in-python-to-find-a-local-minimum-/#gradient-descent-with-a-3d-function

from scipy import misc

import matplotlib.pyplot as plt
import numpy as np
import math

x_global = []
y_global = []
z_global = []


# ----------------------------------------------------------------------------------------#
# Function

def function(x1, x2):
    # return - 1.0 * math.exp(-x1**2 - x2**2);
    return 3*x1 ** 2+10*x2 ** 2;


def distanse(dot1, dot2):
    # given 2 dots in this format: (x1, y1)
    # return the dist
    return math.sqrt(((dot1[0] - dot2[0]) ** 2 + (dot1[1] - dot2[1]) ** 2))


def partial_derivative(func, var=0, point=[]):
    args = point[:]

    def wraps(x):
        args[var] = x
        return func(*args)

    return misc.derivative(wraps, point[var], dx=1e-6)


# def at_the_same_z(x1, x2):
#     alpha = 0.1  # learning rate
#     nb_max_iter = 3000  # Nb max d'iteration
#     eps = 0.0001  # stop condition
#
#     x1_0 = x1  # start point
#     x2_0 = x2
#     z0 = function(x1_0, x2_0)
#     plt.scatter(x1_0, x2_0, c='g')
#
#     cond = eps + 10.0  # start with cond greater than eps (assumption)
#     nb_iter = 0
#     tmp_z0 = z0
#     dots = [(x1_0, x2_0)]
#     score = []
#     while cond > eps and nb_iter < nb_max_iter:
#         partial_derivative_x = partial_derivative(function, 0, [x1_0, x2_0])
#         partial_derivative_y = partial_derivative(function, 1, [x1_0, x2_0])
#         partial_derivative_score = math.sqrt(partial_derivative_x ** 2 + partial_derivative_y ** 2)
#         score.append(partial_derivative_score)
#         tmp_x1_0 = x1_0 - alpha * partial_derivative_y
#         tmp_x2_0 = x2_0 + alpha * partial_derivative_x
#         x1_0 = tmp_x1_0
#         x2_0 = tmp_x2_0
#         z0 = function(x1_0, x2_0)
#         nb_iter = nb_iter + 1
#         cond = abs(tmp_z0 - z0)
#         tmp_z0 = z0
#         dots.append((x1_0, x2_0))
#         # print (x1_0,x2_0,cond)
#         x_global.append(x1_0)
#         y_global.append(x2_0)
#         z_global.append(function(x1_0, x2_0))
#         plt.scatter(x1_0, x2_0, c='g')
#     alpha = 0.1  # learning rate
#     nb_max_iter = 3000  # Nb max d'iteration
#     eps = 0.0001  # stop condition
#
#     x1_0 = x1  # start point
#     x2_0 = x2
#     z0 = function(x1_0, x2_0)
#     plt.scatter(x1_0, x2_0, c='g')
#
#     cond = eps + 10.0  # start with cond greater than eps (assumption)
#     nb_iter = 0
#     tmp_z0 = z0
#     dots = [(x1_0, x2_0)]
#     score = []
#
#     while cond > eps and nb_iter < nb_max_iter:
#         partial_derivative_x = partial_derivative(function, 0, [x1_0, x2_0])
#         partial_derivative_y = partial_derivative(function, 1, [x1_0, x2_0])
#         partial_derivative_score = math.sqrt(partial_derivative_x ** 2 + partial_derivative_y ** 2)
#         score.append(partial_derivative_score)
#         tmp_x1_0 = x1_0 + alpha * partial_derivative_y
#         tmp_x2_0 = x2_0 - alpha * partial_derivative_x
#         x1_0 = tmp_x1_0
#         x2_0 = tmp_x2_0
#         z0 = function(x1_0, x2_0)
#         nb_iter = nb_iter + 1
#         cond = abs(tmp_z0 - z0)
#         tmp_z0 = z0
#         dots.append((x1_0, x2_0))
#         # print (x1_0,x2_0,cond)
#         x_global.append(x1_0)
#         y_global.append(x2_0)
#         z_global.append(function(x1_0, x2_0))
#         plt.scatter(x1_0, x2_0, c='r')


# def at_the_same_z_new_right(x1, x2):
#     alpha = 0.001  # learning rate
#     nb_max_iter = 1000  # Nb max d'iteration
#     eps = 0.01  # stop condition
#
#     x1_0 = x1  # start point
#     x2_0 = x2
#     z0 = function(x1_0, x2_0)
#     plt.scatter(x1_0, x2_0, c='g')
#
#     nb_iter = 0
#     tmp_z0 = z0
#     cond = 0
#     cond_distance = 1
#     temp_dist_from_start_dot = 0
#
#     while cond_distance > 0 and cond < eps and nb_iter < nb_max_iter:
#         partial_derivative_x = partial_derivative(function, 0, [x1_0, x2_0])
#         partial_derivative_y = partial_derivative(function, 1, [x1_0, x2_0])
#         tmp_x1_0 = x1_0 - alpha * partial_derivative_y
#         tmp_x2_0 = x2_0 + alpha * partial_derivative_x
#         x1_0 = tmp_x1_0
#         x2_0 = tmp_x2_0
#         z0 = function(x1_0, x2_0)
#         nb_iter = nb_iter + 1
#         cond = abs(tmp_z0 - z0)
#         # print("Diff" + str(cond))
#         tmp_z0 = z0
#         plt.scatter(x1_0, x2_0, c='g')
#         dist_from_start_dot = distanse((x1_0, x2_0), (x1, x2))
#         cond_distance = dist_from_start_dot - temp_dist_from_start_dot
#         # print("dist" + str(cond_distance))
#         temp_dist_from_start_dot = dist_from_start_dot


def at_the_same_z_new(x1, x2, direction=True, color='g'):
    sign_flag = 1
    if not direction:
        sign_flag = -1
    color = color
    alpha = 0.003  # learning rate
    nb_max_iter = 1000  # Nb max d'iteration
    eps = 0.1  # stop condition

    x1_0 = x1  # start point
    x2_0 = x2
    z0 = function(x1_0, x2_0)
    plt.scatter(x1_0, x2_0, c='g')

    nb_iter = 0
    tmp_z0 = z0
    cond = 0
    cond_distance = 1
    temp_dist_from_start_dot = 0

    while cond_distance > 0 and cond < eps and nb_iter < nb_max_iter:
        partial_derivative_x = partial_derivative(function, 0, [x1_0, x2_0])
        partial_derivative_y = partial_derivative(function, 1, [x1_0, x2_0])
        tmp_x1_0 = x1_0 - 1 * sign_flag * alpha * partial_derivative_y
        tmp_x2_0 = x2_0 + 1 * sign_flag * alpha * partial_derivative_x
        x1_0 = tmp_x1_0
        x2_0 = tmp_x2_0
        z0 = function(x1_0, x2_0)
        nb_iter = nb_iter + 1
        cond = abs(tmp_z0 - z0)
        # print("Diff" + str(cond))
        tmp_z0 = z0
        plt.scatter(x1_0, x2_0, c=color)
        dist_from_start_dot = distanse((x1_0, x2_0), (x1, x2))
        cond_distance = dist_from_start_dot - temp_dist_from_start_dot
        # print("dist" + str(cond_distance))
        temp_dist_from_start_dot = dist_from_start_dot
    if nb_iter >= nb_max_iter:
        print("Reason for stop: nb_iter>=nb_max_ite")  # debug
    if cond_distance <= 0:
        print("Reason for stop: cond_distance <= 0")  # debug
    if cond >= eps:
        print(cond)
        print("Reason for stop: cond >= eps")  # debug


# ----------------------------------------------------------------------------------------#
# Plot Function

x1 = np.arange(-20.0, 20.0, 0.1)
x2 = np.arange(-20.0, 20.0, 0.1)

xx1, xx2 = np.meshgrid(x1, x2);
z=function(xx1, xx2)
h = plt.contourf(x1, x2, z)
#plt.show()

# ----------------------------------------------------------------------------------------#
# Gradient Descent

#learning parameters
alpha = 0.05  # learning rate
nb_max_iter = 300  # Nb max d'iteration
eps = 0.1  # stop condition

x1_0 = 2.0  # start point #to do- add randomness / based on the prior knowledge of the parameters
x2_0 = 2.5
z0 = function(x1_0, x2_0)
plt.scatter(x1_0, x2_0)

cond = eps + 10.0  # start with cond greater than eps (assumption)
nb_iter = 0
tmp_z0 = z0
dots = [(x1_0, x2_0)]
score = []
plt.scatter(x1_0, x2_0, c='violet')
while cond > eps and nb_iter < nb_max_iter:
    partial_derivative_x = partial_derivative(function, 0, [x1_0, x2_0]) #TO DO- calculate partial_derivative with no function
    partial_derivative_y = partial_derivative(function, 1, [x1_0, x2_0]) #TO DO
    partial_derivative_score = math.sqrt(partial_derivative_x ** 2 + partial_derivative_y ** 2)
    score.append(partial_derivative_score)
    tmp_x1_0 = x1_0 - alpha * partial_derivative_x
    tmp_x2_0 = x2_0 - alpha * partial_derivative_y
    x1_0 = tmp_x1_0
    x2_0 = tmp_x2_0
    z0 = function(x1_0, x2_0)
    nb_iter = nb_iter + 1
    cond = abs(tmp_z0 - z0)
    tmp_z0 = z0
    dots.append((x1_0, x2_0))
    plt.scatter(x1_0, x2_0, c='violet')

# print("dots")
# print(dots)
# print("score")
# print(score)

score_histogram = score[:]


while len(score_histogram) >= 1:
    # at each iteration we take the dot with the biggest change and add more dots at the same highest line
    if 1==1 or input("Do you want to add another level")=="yes":
            at_the_same_z_new(dots[score.index(max(score_histogram))][0], dots[score.index(max(score_histogram))][1], True)
            print("dots list")
            print(dots)
            print("score list")
            print(score)
            print("score  histogram list")
            print(score_histogram)
            print("dot")
            print(dots[score.index(max(score_histogram))])
            print("max")
            print(max(score_histogram))
            at_the_same_z_new(dots[score.index(max(score_histogram))][0], dots[score.index(max(score_histogram))][1], False, 'b')
            print("len")
            print(len(score_histogram))
            print(max(score_histogram))
            score_histogram.remove(max(score_histogram))
    else:
        break

plt.title("Gradient Descent Python (2d test)")
plt.savefig("gradiend_descent_2d_python.png", bbox_inches='tight')
plt.show()


