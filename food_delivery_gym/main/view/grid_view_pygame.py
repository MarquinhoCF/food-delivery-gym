import pygame
from food_delivery_gym.main.driver.driver_status import DriverStatus
from food_delivery_gym.main.customer.custumer_status import CustumerStatus
from food_delivery_gym.main.view.food_delivery_view import FoodDeliveryView

# Definindo cores
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
DARK_GREEN = (0, 100, 0)
GREEN = (0, 255, 0)
GRAY = (137, 137, 137)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
DARK_GRAY = (50, 50, 50)
LIGHT_BLUE = (173, 216, 230)
LIGHT_GREEN = (200, 255, 200)
ORANGE = (255, 165, 0)
LIGHT_GRAY = (220, 220, 220)
YELLOW = (255, 255, 200)

def map_coordinate(value, min_val, max_val, min_screen, max_screen):
    return min_screen + (value - min_val) * (max_screen - min_screen) / (max_val - min_val)


class GridViewPygame(FoodDeliveryView):

    def __init__(self, grid_size=50, draw_grid=True, window_size=(1600, 1300), fps=30):
        # Aumenta o painel lateral para acomodar mais informações
        self.side_panel_width = 400
        self.info_panel_height = 180
        self.frame_padding = 20
        
        # Área do mapa principal
        self.map_area_width = window_size[0] - self.side_panel_width - (3 * self.frame_padding)
        self.map_area_height = window_size[1] - self.info_panel_height - (2 * self.frame_padding)
        
        super().__init__(grid_size, window_size, fps)
        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode(window_size)
        self.clock = pygame.time.Clock()
        self.draw_grid = draw_grid
        pygame.display.set_caption('Food Delivery Simulation - Map View')
        
        # Fontes para texto
        self.font_title = pygame.font.SysFont('Arial', 24, bold=True)
        self.font_info = pygame.font.SysFont('Arial', 20)
        self.font_small = pygame.font.SysFont('Arial', 16)
        self.font_id = pygame.font.SysFont('Arial', 12, bold=True)
        self.font_coord = pygame.font.SysFont('Arial', 13)
        self.font_panel = pygame.font.SysFont('Arial', 14)
        self.font_panel_small = pygame.font.SysFont('Arial', 13)
        
        # === OTIMIZAÇÕES: Cache de superfícies estáticas ===
        self.static_background = None
        self.establishments_layer = None
        self.panels_background = None
        
        # Cache de dados para detectar mudanças
        self.last_driver_count = 0
        self.last_driver_states = {}
        self.last_establishment_count = 0
        self.last_stats = {}
        
        # Cache de textos renderizados
        self.text_cache = {}
        
        # Cache para coordenadas dos estabelecimentos
        self.establishments_coords_cache = None
        
        # Superfícies reutilizáveis para elementos dinâmicos
        self.dynamic_layer = pygame.Surface((self.map_area_width, self.map_area_height), pygame.SRCALPHA)
        
        # Cache para painéis laterais e info
        self.info_panel_cache = None
        self.drivers_panel_cache = None
        self.establishments_panel_cache = None
        
        # Recortes de tela (dirty rects) para atualização parcial
        self.dirty_rects = []
        
    def get_cached_text(self, text, font, color):
        """Cache de textos renderizados para evitar re-renderização"""
        cache_key = (text, id(font), color)
        if cache_key not in self.text_cache:
            self.text_cache[cache_key] = font.render(text, True, color)
        return self.text_cache[cache_key]

    def coordinate(self, coordinate):
        return (
            self.frame_padding + map_coordinate(coordinate[0], self.min_x, self.max_x, 0, self.map_area_width),
            self.frame_padding + map_coordinate(coordinate[1], self.min_y, self.max_y, 0, self.map_area_height)
        )
    
    def create_static_background(self):
        """Cria a superfície estática com grid, frame e coordenadas (chamado uma vez)"""
        if self.static_background is not None:
            return self.static_background
            
        surface = pygame.Surface(self.window_size)
        surface.fill(WHITE)
        
        # Desenha a moldura
        frame_rect = pygame.Rect(
            self.frame_padding - 2,
            self.frame_padding - 2,
            self.map_area_width + 4,
            self.map_area_height + 4
        )
        pygame.draw.rect(surface, DARK_GRAY, frame_rect, 3)
        
        # Desenha o grid se habilitado
        if self.draw_grid:
            # Grid lines
            for i in range(self.grid_size + 1):
                x = self.frame_padding + map_coordinate(i, 0, self.grid_size, 0, self.map_area_width)
                y = self.frame_padding + map_coordinate(i, 0, self.grid_size, 0, self.map_area_height)
                pygame.draw.line(surface, GRAY, (x, self.frame_padding), (x, self.frame_padding + self.map_area_height), 1)
                pygame.draw.line(surface, GRAY, (self.frame_padding, y), (self.frame_padding + self.map_area_width, y), 1)
            
            # Coordenadas do grid
            step = 5
            for i in range(0, self.grid_size + 1, step):
                x = self.frame_padding + map_coordinate(i, 0, self.grid_size, 0, self.map_area_width)
                y = self.frame_padding + map_coordinate(i, 0, self.grid_size, 0, self.map_area_height)
                
                coord_text = self.font_coord.render(str(i), True, DARK_GRAY)
                text_rect = coord_text.get_rect(center=(x, self.frame_padding - 8))
                surface.blit(coord_text, text_rect)
                
                coord_text = self.font_coord.render(str(i), True, DARK_GRAY)
                text_rect = coord_text.get_rect(center=(self.frame_padding - 10, y))
                surface.blit(coord_text, text_rect)
        
        self.static_background = surface
        return surface
    
    def create_establishments_layer(self, environment):
        """Cria camada dos estabelecimentos (não muda)"""
        if self.establishments_layer is not None:
            return self.establishments_layer
            
        surface = pygame.Surface((self.map_area_width, self.map_area_height), pygame.SRCALPHA)
        
        # Cache das coordenadas
        if self.establishments_coords_cache is None:
            self.establishments_coords_cache = [
                (est.establishment_id, est.coordinate, 
                 hasattr(est, "operating_radius") and est.operating_radius or None) 
                for est in environment.state.establishments
            ]
        
        for est_id, coordinate, operating_radius in self.establishments_coords_cache:
            mapped_x, mapped_y = self.coordinate(coordinate)
            # Ajustar para coordenadas relativas à camada
            rel_x = mapped_x - self.frame_padding
            rel_y = mapped_y - self.frame_padding
            
            # Corpo do restaurante
            house_size = 15
            pygame.draw.rect(surface, GREEN, (rel_x - house_size // 2, rel_y - house_size // 2, house_size, house_size))
            
            # Telhado
            pygame.draw.polygon(surface, GREEN, [
                (rel_x, rel_y - house_size),
                (rel_x - house_size // 2, rel_y - house_size // 2),
                (rel_x + house_size // 2, rel_y - house_size // 2)
            ])
            
            # ID do estabelecimento
            id_text = self.font_id.render(str(est_id), True, WHITE)
            id_rect = id_text.get_rect(center=(rel_x, rel_y - 2))
            surface.blit(id_text, id_rect)
            
            # Raio de operação
            if operating_radius:
                operating_radius_mapped = map_coordinate(operating_radius, 0, 100, 0, min(self.map_area_width, self.map_area_height))
                pygame.draw.circle(surface, GREEN, (int(rel_x), int(rel_y)), int(operating_radius_mapped), 1)
        
        self.establishments_layer = surface
        self.last_establishment_count = len(environment.state.establishments)
        return surface
    
    def draw_driver(self, canvas, driver_color, x, y, driver_id):
        """Desenha motorista em coordenadas relativas"""
        car_length = 18
        car_height = 6
        pygame.draw.rect(canvas, driver_color, (x - car_length // 2, y - car_height // 2, car_length, car_height), border_radius=3)
        
        wheel_radius = 2
        pygame.draw.circle(canvas, BLACK, (x - car_length // 3, y + car_height // 2), wheel_radius)
        pygame.draw.circle(canvas, BLACK, (x + car_length // 3, y + car_height // 2), wheel_radius)
        
        top_width = 8
        top_height = 4.5
        pygame.draw.rect(canvas, BLACK, (x - top_width // 2, y - top_height, top_width, top_height))

    def draw_customer(self, canvas, x, y):
        """Desenha cliente em coordenadas relativas"""
        pin_length = 7
        circle_radius = 5
        
        pygame.draw.line(canvas, BLUE, (x, y), (x, y - pin_length), 2)
        pygame.draw.circle(canvas, BLUE, (int(x), int(y - pin_length - circle_radius)), circle_radius)

    def draw_dynamic_elements(self, environment):
        """Desenha apenas elementos que mudam (motoristas, clientes, rotas)"""
        # Limpa a camada dinâmica
        self.dynamic_layer.fill((0, 0, 0, 0))
        
        # Desenhar clientes
        for customer in environment.state.customers:
            if customer.status == CustumerStatus.WAITING_DELIVERY:
                mapped_x, mapped_y = self.coordinate(customer.coordinate)
                rel_x = mapped_x - self.frame_padding
                rel_y = mapped_y - self.frame_padding
                self.draw_customer(self.dynamic_layer, rel_x, rel_y)
        
        # Desenhar motoristas e rotas
        for driver in environment.state.drivers:
            mapped_x, mapped_y = self.coordinate(driver.coordinate)
            rel_x = mapped_x - self.frame_padding
            rel_y = mapped_y - self.frame_padding
            
            # Rota
            if driver.status in [DriverStatus.PICKING_UP, DriverStatus.PICKING_UP_WAITING, 
                                DriverStatus.DELIVERING, DriverStatus.DELIVERING_WAITING]:
                target_mapped_x, target_mapped_y = self.coordinate(driver.current_route_segment.coordinate)
                target_rel_x = target_mapped_x - self.frame_padding
                target_rel_y = target_mapped_y - self.frame_padding
                pygame.draw.line(self.dynamic_layer, RED, (rel_x, rel_y), (target_rel_x, target_rel_y), 2)
            
            # Motorista
            self.draw_driver(self.dynamic_layer, driver.color, rel_x, rel_y, driver.driver_id)
        
        return self.dynamic_layer

    def get_driver_state_hash(self, driver):
        """Gera hash do estado do motorista para detectar mudanças"""
        route_info = None
        if driver.current_route_segment:
            route_info = (
                driver.current_route_segment.coordinate,
                driver.current_route_segment.route_segment_type.name,
                driver.current_route_segment.order.order_id,
                driver.current_route_segment.order.establishment.establishment_id,
                driver.current_route_segment.order.customer.coordinate
            )
        
        return (
            driver.driver_id,
            driver.coordinate,
            driver.status.name,
            len(driver.orders_list),
            route_info
        )

    def draw_drivers_panel(self, canvas, environment, force_redraw=False):
        # Gera hash do estado atual de todos os motoristas
        current_states = {d.driver_id: self.get_driver_state_hash(d) for d in environment.state.drivers}
        
        # Verifica se precisa redesenhar (comparando estados)
        if not force_redraw and current_states == self.last_driver_states:
            if self.drivers_panel_cache:
                return
        
        panel_x = self.frame_padding + self.map_area_width + self.frame_padding
        panel_y = self.frame_padding
        panel_width = self.side_panel_width
        panel_height = round(self.map_area_height * 0.85) - self.frame_padding
        
        # Fundo do painel
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(canvas, LIGHT_BLUE, panel_rect)
        pygame.draw.rect(canvas, DARK_GRAY, panel_rect, 2)
        
        # Título
        title_text = self.font_info.render("Drivers", True, DARK_GRAY)
        canvas.blit(title_text, (panel_x + 10, panel_y + 10))
        
        pygame.draw.line(canvas, DARK_GRAY, (panel_x + 5, panel_y + 35), 
                        (panel_x + panel_width - 5, panel_y + 35), 1)
        
        y_offset = panel_y + 55
        line_height = 70  # Aumentado para acomodar mais informações
        max_visible = (panel_height - 50) // line_height
        
        drivers = environment.state.drivers[:max_visible]
        
        for driver in drivers:
            if y_offset + line_height > panel_y + panel_height - 5:
                break
            
            # Fundo para cada motorista
            driver_bg_rect = pygame.Rect(panel_x + 5, y_offset - 3, panel_width - 10, line_height - 5)
            pygame.draw.rect(canvas, YELLOW if driver.status != DriverStatus.AVAILABLE else WHITE, driver_bg_rect, 0, border_radius=5)
            pygame.draw.rect(canvas, DARK_GRAY, driver_bg_rect, 1, border_radius=5)
            
            # ID e Status
            id_text = self.font_panel.render(f"ID: {driver.driver_id} | Status: {driver.status.name}", True, DARK_GRAY)
            canvas.blit(id_text, (panel_x + 10, y_offset))
            
            # Posição atual
            coord_text = self.font_panel_small.render(
                f"Pos: ({driver.coordinate[0]:.1f}, {driver.coordinate[1]:.1f})", 
                True, DARK_GRAY
            )
            canvas.blit(coord_text, (panel_x + 10, y_offset + 16))
            
            # Informações condicionais baseadas no estado
            info_y = y_offset + 30
            
            if driver.current_route_segment:
                segment = driver.current_route_segment
                order = segment.order
                
                # Tipo de segmento
                segment_type = "PICKUP" if segment.is_pickup() else "DELIVERY"
                segment_color = ORANGE if segment.is_pickup() else BLUE

                type_text = self.font_panel_small.render(f"Route Segment:    Type: {segment_type}   Destination: ({segment.coordinate[0]:.1f}, {segment.coordinate[1]:.1f})", True, segment_color)
                canvas.blit(type_text, (panel_x + 10, info_y))
                
                # ID do Estabelecimento
                est_text = self.font_panel_small.render(
                    f"Current Order Info:  Est. ID: {order.establishment.establishment_id}    Customer Pos.: ({order.customer.coordinate[0]:.1f}, {order.customer.coordinate[1]:.1f})",
                    True, DARK_GRAY
                )
                canvas.blit(est_text, (panel_x + 10, info_y + 16))
                
            elif len(driver.orders_list) > 0:
                # Motorista tem pedidos mas não está em rota ativa
                pending_text = self.font_panel_small.render(
                    f"Pending Orders: {len(driver.orders_list)}",
                    True, ORANGE
                )
                canvas.blit(pending_text, (panel_x + 10, info_y))
            else:
                # Motorista disponível
                available_text = self.font_panel_small.render(
                    "Waiting for order...",
                    True, DARK_GREEN
                )
                canvas.blit(available_text, (panel_x + 10, info_y))
            
            # Círculo colorido indicador
            pygame.draw.circle(canvas, driver.color, (panel_x + panel_width - 20, y_offset + 20), 8)
            
            y_offset += line_height
        
        self.last_driver_states = current_states
    
    def draw_establishments_panel(self, canvas, environment):
        """Desenha painel com informações dos estabelecimentos"""
        panel_x = self.frame_padding + self.map_area_width + self.frame_padding
        panel_y = self.frame_padding + round(self.map_area_height * 0.85)
        panel_width = self.side_panel_width
        panel_height = round(self.map_area_height * 0.35) - (2 * self.frame_padding) + 4
        
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(canvas, LIGHT_GREEN, panel_rect)
        pygame.draw.rect(canvas, DARK_GRAY, panel_rect, 2)
        
        title_text = self.font_info.render("Establishments", True, DARK_GRAY)
        canvas.blit(title_text, (panel_x + 10, panel_y + 10))
        
        pygame.draw.line(canvas, DARK_GRAY, (panel_x + 5, panel_y + 35), 
                        (panel_x + panel_width - 5, panel_y + 35), 1)
        
        # Usar cache separado para o painel
        panel_cache = [(est_id, coord) for est_id, coord, _ in self.establishments_coords_cache] if self.establishments_coords_cache else []
        
        if not panel_cache:
            panel_cache = [
                (est.establishment_id, est.coordinate) 
                for est in environment.state.establishments
            ]
        
        y_offset = panel_y + 45
        line_height = 25
        max_visible = (panel_height - 50) // line_height
        
        establishments = panel_cache[:max_visible]
        
        for est_id, coordinate in establishments:
            if y_offset + line_height > panel_y + panel_height - 5:
                break

            id_and_coordinate_text = self.font_panel.render(f"ID: {est_id}  -->  Pos: ({coordinate[0]:.1f}, {coordinate[1]:.1f})", True, DARK_GRAY)
            canvas.blit(id_and_coordinate_text, (panel_x + 10, y_offset))
            
            pygame.draw.circle(canvas, GREEN, (panel_x + panel_width - 20, y_offset + 12), 6)
            
            y_offset += line_height

    def draw_info_panel(self, canvas, environment):
        """Desenha painel de informações inferior"""
        panel_y = self.frame_padding + self.map_area_height + 20
        panel_height = self.info_panel_height - 20
        
        # Coletar estatísticas atuais
        current_stats = {
            'time': int(environment.now),
            'delivered': environment.state.orders_delivered,
            'waiting': sum(1 for c in environment.state.customers if c.status == CustumerStatus.WAITING_DELIVERY),
            'available': sum(1 for d in environment.state.drivers if d.status == DriverStatus.AVAILABLE),
            'picking': sum(1 for d in environment.state.drivers if d.status == DriverStatus.PICKING_UP),
            'delivering': sum(1 for d in environment.state.drivers if d.status == DriverStatus.DELIVERING),
            'total_drivers': len(environment.state.drivers)
        }
        
        # Só redesenha se mudou
        if current_stats == self.last_stats and self.info_panel_cache is not None:
            canvas.blit(self.info_panel_cache, (self.frame_padding, panel_y))
            return
        
        # Criar nova superfície para o painel
        panel_surface = pygame.Surface((self.map_area_width, panel_height))
        
        panel_rect = pygame.Rect(0, 0, self.map_area_width, panel_height)
        pygame.draw.rect(panel_surface, LIGHT_BLUE, panel_rect)
        pygame.draw.rect(panel_surface, DARK_GRAY, panel_rect, 2)
        
        title_text = self.font_title.render("Simulation Statistics", True, DARK_GRAY)
        panel_surface.blit(title_text, (20, 10))
        
        pygame.draw.line(panel_surface, DARK_GRAY, 
                        (10, 45), 
                        (self.map_area_width - 10, 45), 1)
        
        col1_x = 30
        info_y = 60
        line_spacing = 25
        
        info_texts = [
            f"Time: {current_stats['time']}",
            f"Orders Delivered: {current_stats['delivered']}",
            f"Orders Waiting: {current_stats['waiting']}",
        ]
        
        for i, text in enumerate(info_texts):
            rendered_text = self.font_info.render(text, True, DARK_GRAY)
            panel_surface.blit(rendered_text, (col1_x, info_y + i * line_spacing))
        
        col2_x = self.map_area_width // 2 + 30
        
        info_texts_col2 = [
            f"Drivers Available: {current_stats['available']}/{current_stats['total_drivers']}",
            f"Picking Up: {current_stats['picking']}  |  Delivering: {current_stats['delivering']}",
        ]
        
        for i, text in enumerate(info_texts_col2):
            rendered_text = self.font_info.render(text, True, DARK_GRAY)
            panel_surface.blit(rendered_text, (col2_x, info_y + i * line_spacing))
        
        # Atualizar cache
        self.info_panel_cache = panel_surface
        self.last_stats = current_stats
        
        # Desenhar no canvas principal
        canvas.blit(self.info_panel_cache, (self.frame_padding, panel_y))

    def render(self, environment):
        """Método principal de renderização otimizado"""
        self.quited = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quited = True

        if self.quited:
            return

        # 1. Criar/recuperar fundo estático (apenas uma vez)
        static_bg = self.create_static_background()
        self.screen.blit(static_bg, (0, 0))
        
        # 2. Criar/recuperar camada de estabelecimentos (raramente muda)
        establishments = self.create_establishments_layer(environment)
        self.screen.blit(establishments, (self.frame_padding, self.frame_padding))
        
        # 3. Desenhar elementos dinâmicos (motoristas, clientes, rotas)
        dynamic = self.draw_dynamic_elements(environment)
        self.screen.blit(dynamic, (self.frame_padding, self.frame_padding))
        
        # 4. Desenhar painéis (otimizados com cache)
        self.draw_drivers_panel(self.screen, environment)
        self.draw_establishments_panel(self.screen, environment)
        self.draw_info_panel(self.screen, environment)

        # 5. Atualizar display
        pygame.display.flip()
        self.clock.tick(self.fps)

    def quit(self):
        # Limpar cache
        self.text_cache.clear()
        self.static_background = None
        self.establishments_layer = None
        self.dynamic_layer = None
        self.info_panel_cache = None
        self.drivers_panel_cache = None
        self.establishments_panel_cache = None
        
        pygame.display.quit()
        pygame.quit()