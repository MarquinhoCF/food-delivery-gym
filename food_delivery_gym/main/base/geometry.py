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
    r = abs(rng_generator.normal(loc=0, scale=radius))  # gauss
    x = centroid[0] + r * math.cos(theta)
    y = centroid[1] + r * math.sin(theta)
    return round(x), round(y)


# def point_in_gauss_circle(centroid, radius, limit, rng_generator: RandomManager):
#     while True:
#         x, y = point_in_gauss_radius(centroid, radius, rng_generator)
#         if 0 <= x <= limit and 0 <= y <= limit:
#             return x, y


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

def point_in_gauss_circle(
    centroid_node: int,
    radius_meters: float,
    osmnx_map,           # OSMnxMap
    rng: RandomManager,
) -> int:
    """
    Gera um node_id dentro de ~radius_meters do centroid_node,
    usando distribuição gaussiana e snap para a rua mais próxima.
    """
    lat_c, lon_c = osmnx_map.to_latlon(centroid_node)

    # Converte raio de metros para graus (aproximação)
    deg_per_m = 1 / 111_000

    for _ in range(50):   # tentativas
        theta = rng.uniform(0, 2 * math.pi)
        r = abs(rng.normal(0, radius_meters * deg_per_m))
        lat = lat_c + r * math.cos(theta)
        lon = lon_c + r * math.sin(theta)
        node = osmnx_map.nearest_node(lat, lon)
        if node in osmnx_map._node_data:
            return node

    return centroid_node   # fallback