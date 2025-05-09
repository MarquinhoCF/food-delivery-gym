from typing import List

from food_delivery_gym.main.base.types import Coordinate, Number
from food_delivery_gym.main.map.map import Map


class GridMap(Map):
    def __init__(self, size):
        super().__init__(size)
        self.generated_points = {}

    def distance(self, coord1: Coordinate, coord2: Coordinate) -> Number:
        return max(abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1]), 1)

    def max_distance(self) -> Number:
        # abs(size-1 - 0) + abs(size-1 - 0) = (size-1) + (size-1) = 2*size - 2
        return self.size * 2 - 2

    def acc_distance(self, coordinates: List[Coordinate]) -> Number:
        distance = 0
        if len(coordinates) <= 1:
            return distance

        previous_coordinate = coordinates[0]
        for coordinate in coordinates[1:]:
            distance += self.distance(previous_coordinate, coordinate)
            previous_coordinate = coordinate

        return distance

    def estimated_time(self, coord1: Coordinate, coord2: Coordinate, rate: Number) -> Number:
        return round(self.distance(coord1, coord2) / rate)

    def random_point(self, not_repeated=False) -> Coordinate:
        point = self.rng.randrange(self.size), self.rng.randrange(self.size)
        if not_repeated:
            while point in self.generated_points:
                point = self.rng.randrange(self.size), self.rng.randrange(self.size)
            self.generated_points[point] = True
        return point

    def move(self, origin: Coordinate, destination: Coordinate, rate: Number) -> Coordinate:
        x, y = origin
        dest_x, dest_y = destination

        # Calcula a diferença absoluta entre as coordenadas de origem e destino no eixo X, Y (horizontal, vertical)
        dx = abs(dest_x - x)
        dy = abs(dest_y - y)
        
        # Arredonda a taxa de movimento
        rate = round(rate)
        # Calcula a metade da taxa
        rate2 = round(rate / 2)

        # Verifica se a soma das diferenças no eixo X e Y é menor ou igual à taxa de movimento
        if (dx + dy) <= rate:
            # Se sim, o destino é alcançado diretamente
            return dest_x, dest_y
        else:
            # Caso contrário, realiza o movimento em duas etapas (X e Y)

            # Se a diferença no eixo X for menor que no eixo Y
            if dx < dy:
                # A distância a ser percorrida no eixo X será o menor valor entre a diferença X e metade da taxa de movimento
                distance = min(dx, rate - rate2)

                # Move na direção X dependendo de qual lado o destino está
                if x < dest_x:
                    x += distance
                else:
                    x -= distance

                # No eixo Y, move o restante da taxa
                if y < dest_y:
                    y += rate - distance
                else:
                    y -= rate - distance

            # Se a diferença no eixo Y for menor ou igual ao eixo X
            else:
                # A distância a ser percorrida no eixo Y será o menor valor entre a diferença Y e metade da taxa de movimento
                distance = min(dy, rate - rate2)

                # Move na direção Y dependendo de qual lado o destino está
                if y < dest_y:
                    y += distance
                else:
                    y -= distance

                # No eixo X, move o restante da taxa
                if x < dest_x:
                    x += rate - distance
                else:
                    x -= rate - distance

        return x, y

