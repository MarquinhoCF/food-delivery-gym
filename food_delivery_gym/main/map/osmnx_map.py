from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import osmnx as ox

from food_delivery_gym.main.base.types import Coordinate, Number
from food_delivery_gym.main.map.map import Map
from food_delivery_gym.main.utils.random_manager import RandomManager


# Lavras, MG — bounding box do perímetro urbano (ajuste se necessário)
LAVRAS_URBAN_BBOX = {
    "north": -21.218,
    "south": -21.265,
    "east":  -44.970,
    "west":  -45.010,
}

# Coordinate aqui é sempre (node_id,) internamente, mas exposto como (lat, lon)
# Usamos node_id como chave primária e (lat,lon) só para display/WebSocket

class OSMnxMap(Map):
    """
    Mapa real baseado na malha viária de Lavras, MG.

    Internamente, 'Coordinate' é um inteiro OSMnx node_id.
    Métodos de interface recebem e devolvem node_ids,
    mas expõem lat/lon via to_latlon().
    """

    CACHE_PATH = Path("data/map_cache/lavras_urban.pkl")

    def __init__(self, cache: bool = True):
        # size é usado por alguns módulos como referência — mantemos
        super().__init__(size=1)
        self.rng = RandomManager().get_random_instance()
        self.G: nx.MultiDiGraph = self._load_or_download(cache)
        self._nodes = list(self.G.nodes())
        self._node_data = {n: d for n, d in self.G.nodes(data=True)}

        # cache de distâncias para pares frequentes
        self._dist_cache: dict[tuple, float] = {}

    # ------------------------------------------------------------------
    # Download / cache
    # ------------------------------------------------------------------

    def _load_or_download(self, cache: bool) -> nx.MultiDiGraph:
        if cache and self.CACHE_PATH.exists():
            with open(self.CACHE_PATH, "rb") as f:
                return pickle.load(f)

        print("Baixando malha viária de Lavras (OSMnx)…")
        G = ox.graph_from_bbox(
            bbox=(
                LAVRAS_URBAN_BBOX["west"],
                LAVRAS_URBAN_BBOX["south"],
                LAVRAS_URBAN_BBOX["east"],
                LAVRAS_URBAN_BBOX["north"],
            ),
            network_type="drive",
            simplify=False,
        )
        # Adiciona velocidades e tempos de viagem
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)

        if cache:
            self.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(self.CACHE_PATH, "wb") as f:
                pickle.dump(G, f)

        return G

    # ------------------------------------------------------------------
    # Conversões
    # ------------------------------------------------------------------

    def to_latlon(self, node_id: int) -> Tuple[float, float]:
        """Devolve (lat, lon) de um node_id."""
        d = self._node_data[node_id]
        return d["y"], d["x"]   # y=lat, x=lon no OSMnx

    def nearest_node(self, lat: float, lon: float) -> int:
        """Snap de coordenada geográfica para o nó mais próximo da rede."""
        return ox.distance.nearest_nodes(self.G, X=lon, Y=lat)

    # ------------------------------------------------------------------
    # Interface Map
    # ------------------------------------------------------------------

    def distance(self, coord1: int, coord2: int) -> Number:
        """Distância de rota em metros entre dois node_ids."""
        if coord1 == coord2:
            return 0.0
        key = (coord1, coord2)
        if key not in self._dist_cache:
            try:
                length = nx.shortest_path_length(
                    self.G, coord1, coord2, weight="length"
                )
            except nx.NetworkXNoPath:
                # Fallback: distância euclidiana em metros
                lat1, lon1 = self.to_latlon(coord1)
                lat2, lon2 = self.to_latlon(coord2)
                length = self._haversine(lat1, lon1, lat2, lon2)
            self._dist_cache[key] = max(1.0, length)
        return self._dist_cache[key]

    def acc_distance(self, coordinates: List[int]) -> Number:
        total = 0.0
        for a, b in zip(coordinates[:-1], coordinates[1:]):
            total += self.distance(a, b)
        return total

    def estimated_time(
        self, coord1: int, coord2: int, rate: Number
    ) -> Number:
        """
        Tempo estimado em segundos de simulação.
        'rate' agora é velocidade média em m/s (ex: 8.3 ≈ 30 km/h).
        """
        if coord1 == coord2:
            return 0
        dist = self.distance(coord1, coord2)
        return max(1, math.ceil(dist / rate))

    def random_point(self, not_repeated: bool = False) -> int:
        """Retorna um node_id aleatório da rede viária."""
        idx = self.rng.integers(0, len(self._nodes))
        return self._nodes[int(idx)]

    def move(
        self, origin: int, destination: int, rate: Number
    ) -> int:
        """
        Avança 'rate' metros ao longo do caminho mais curto.
        Devolve o node_id mais próximo da posição após mover.
        """
        if origin == destination:
            return destination
        try:
            path = nx.shortest_path(
                self.G, origin, destination, weight="length"
            )
        except nx.NetworkXNoPath:
            return destination

        accumulated = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_data = min(
                self.G[u][v].values(), key=lambda d: d.get("length", 0)
            )
            edge_len = edge_data.get("length", 1.0)

            if accumulated + edge_len >= rate:
                # Ficamos neste segmento
                return v   # simplificação: snap ao próximo nó
            accumulated += edge_len

        return path[-1]

    def max_distance(self) -> Number:
        """Diagonal aproximada da bbox em metros."""
        lat1, lon1 = LAVRAS_URBAN_BBOX["south"], LAVRAS_URBAN_BBOX["west"]
        lat2, lon2 = LAVRAS_URBAN_BBOX["north"], LAVRAS_URBAN_BBOX["east"]
        return self._haversine(lat1, lon1, lat2, lon2)

    def get_path_latlon(self, origin: int, destination: int) -> List[Tuple[float, float]]:
        """Retorna lista de (lat, lon) do caminho — usado pela visualização."""
        try:
            path = nx.shortest_path(self.G, origin, destination, weight="length")
        except nx.NetworkXNoPath:
            return [self.to_latlon(origin), self.to_latlon(destination)]
        return [self.to_latlon(n) for n in path]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2) -> float:
        R = 6_371_000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))