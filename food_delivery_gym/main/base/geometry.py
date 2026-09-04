import math

import numpy as np


def random_point_in_radius(centroid, inf_limit, sup_limit, rng: np.random.Generator):
    theta = rng.uniform(0, 2 * math.pi)
    r = rng.uniform(inf_limit, sup_limit)
    x = centroid[0] + r * math.cos(theta)
    y = centroid[1] + r * math.sin(theta)
    return round(x), round(y)


def point_in_gauss_radius(centroid, radius, rng: np.random.Generator):
    theta = rng.uniform(0, 2 * math.pi)
    r = abs(rng.normal(loc=0, scale=radius))  # gauss
    x = centroid[0] + r * math.cos(theta)
    y = centroid[1] + r * math.sin(theta)
    return round(x), round(y)


def point_in_gauss_circle(centroid, radius, limit, rng: np.random.Generator):
    while True:
        x, y = point_in_gauss_radius(centroid, radius, rng)
        if 0 <= x <= limit and 0 <= y <= limit:
            return x, y


def random_point_in_circle(centroid, radius, limit, rng: np.random.Generator):
    while True:
        x, y = random_point_in_radius(centroid, 0, radius, rng)
        if 0 <= x <= limit and 0 <= y <= limit:
            return x, y


def random_point_outside_circle(centroid, radius, limit, rng: np.random.Generator):
    while True:
        x, y = random_point_in_radius(centroid, radius + 1, limit, rng)
        if 0 <= x <= limit and 0 <= y <= limit:
            return x, y
