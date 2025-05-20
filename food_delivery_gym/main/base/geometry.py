import math
from food_delivery_gym.main.utils.random_manager import RandomManager

def random_point_in_radius(centroid, inf_limit, sup_limit, rng_generator: RandomManager):
    theta = rng_generator.uniform(0, 2 * math.pi)
    r = rng_generator.uniform(inf_limit, sup_limit)
    x = centroid[0] + r * math.cos(theta)
    y = centroid[1] + r * math.sin(theta)
    return round(x), round(y)


def point_in_gauss_radius(centroid, radius, rng_generator: RandomManager):
    theta = rng_generator.uniform(0, 2 * math.pi)
    r = abs(rng_generator.gauss(0, radius))
    x = centroid[0] + r * math.cos(theta)
    y = centroid[1] + r * math.sin(theta)
    return round(x), round(y)


def point_in_gauss_circle(centroid, radius, limit, rng_generator: RandomManager):
    while True:
        x, y = point_in_gauss_radius(centroid, radius, rng_generator)
        if 0 <= x <= limit and 0 <= y <= limit:
            return x, y


def random_point_in_circle(centroid, radius, limit, rng_generator: RandomManager):
    while True:
        x, y = random_point_in_radius(centroid, 0, radius, rng_generator)
        if 0 <= x <= limit and 0 <= y <= limit:
            return x, y


def random_point_outside_circle(centroid, radius, limit, rng_generator: RandomManager):
    while True:
        x, y = random_point_in_radius(centroid, radius + 1, limit, rng_generator)
        if 0 <= x <= limit and 0 <= y <= limit:
            return x, y
