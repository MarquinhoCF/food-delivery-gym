import pygame

from food_delivery_gym.main.driver.driver_status import DriverStatus
from food_delivery_gym.main.customer.custumer_status import CustumerStatus
from food_delivery_gym.main.view.food_delivery_view import FoodDeliveryView

# Definindo cores
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GRAY = (137, 137, 137)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
DARK_GRAY = (50, 50, 50)
LIGHT_BLUE = (173, 216, 230)
LIGHT_GREEN = (200, 255, 200)
ORANGE = (255, 165, 0)
LIGHT_GRAY = (220, 220, 220)

def map_coordinate(value, min_val, max_val, min_screen, max_screen):
    return min_screen + (value - min_val) * (max_screen - min_screen) / (max_val - min_val)


class GridViewPygame(FoodDeliveryView):

    def __init__(self, grid_size=50, draw_grid=True, window_size=(1600, 1300), fps=30):
        # Adiciona espaço para painéis laterais
        self.side_panel_width = 300
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
        self.font_info = pygame.font.SysFont('Arial', 18)
        self.font_small = pygame.font.SysFont('Arial', 14)
        self.font_id = pygame.font.SysFont('Arial', 10, bold=True)
        self.font_coord = pygame.font.SysFont('Arial', 9)
        self.font_panel = pygame.font.SysFont('Arial', 12)
        
        # Cache para coordenadas dos estabelecimentos (não mudam)
        self.establishments_coords_cache = None

    def coordinate(self, coordinate):
        return (
            self.frame_padding + map_coordinate(coordinate[0], self.min_x, self.max_x, 0, self.map_area_width),
            self.frame_padding + map_coordinate(coordinate[1], self.min_y, self.max_y, 0, self.map_area_height)
        )
    
    def draw_background_grid(self, canvas, color=GRAY):
        # Desenha linhas do grid
        for i in range(self.grid_size + 1):
            x = self.frame_padding + map_coordinate(i, 0, self.grid_size, 0, self.map_area_width)
            y = self.frame_padding + map_coordinate(i, 0, self.grid_size, 0, self.map_area_height)

            pygame.draw.line(canvas, color, (x, self.frame_padding), (x, self.frame_padding + self.map_area_height), 1)
            pygame.draw.line(canvas, color, (self.frame_padding, y), (self.frame_padding + self.map_area_width, y), 1)
    
    def draw_grid_coordinates(self, canvas):
        """Desenha as coordenadas do grid nas bordas"""
        coord_color = DARK_GRAY
        
        # Intervalo para mostrar coordenadas (a cada 5 unidades para não poluir)
        step = 5
        
        for i in range(0, self.grid_size + 1, step):
            # Posição na tela
            x = self.frame_padding + map_coordinate(i, 0, self.grid_size, 0, self.map_area_width)
            y = self.frame_padding + map_coordinate(i, 0, self.grid_size, 0, self.map_area_height)
            
            # Coordenadas horizontais (eixo X) - na parte superior
            coord_text = self.font_coord.render(str(i), True, coord_color)
            text_rect = coord_text.get_rect(center=(x, self.frame_padding - 8))
            canvas.blit(coord_text, text_rect)
            
            # Coordenadas verticais (eixo Y) - na lateral esquerda
            coord_text = self.font_coord.render(str(i), True, coord_color)
            text_rect = coord_text.get_rect(center=(self.frame_padding - 10, y))
            canvas.blit(coord_text, text_rect)
    
    def draw_frame(self, canvas):
        # Desenha moldura ao redor da área do mapa
        frame_rect = pygame.Rect(
            self.frame_padding - 2,
            self.frame_padding - 2,
            self.map_area_width + 4,
            self.map_area_height + 4
        )
        pygame.draw.rect(canvas, DARK_GRAY, frame_rect, 3)
    
    def draw_driver(self, canvas, driver_color, mapped_x, mapped_y, driver_id):
        # Corpo do carro
        car_length = 18
        car_height = 6
        pygame.draw.rect(canvas, driver_color, (mapped_x - car_length // 2, mapped_y - car_height // 2, car_length, car_height), border_radius=3)

        # Rodas do carro
        wheel_radius = 2
        pygame.draw.circle(canvas, BLACK, (mapped_x - car_length // 3, mapped_y + car_height // 2), wheel_radius)
        pygame.draw.circle(canvas, BLACK, (mapped_x + car_length // 3, mapped_y + car_height // 2), wheel_radius)

        # Teto do carro
        top_width = 8
        top_height = 4.5
        pygame.draw.rect(canvas, BLACK, (mapped_x - top_width // 2, mapped_y - top_height, top_width, top_height))
        
        # Desenha o ID do motorista
        id_text = self.font_id.render(str(driver_id), True, WHITE)
        id_rect = id_text.get_rect(center=(mapped_x, mapped_y))
        canvas.blit(id_text, id_rect)

    def draw_establishment(self, canvas, mapped_x, mapped_y, establishment_id):
        # Corpo do restaurante
        house_size = 15
        pygame.draw.rect(canvas, GREEN, (mapped_x - house_size // 2, mapped_y - house_size // 2, house_size, house_size))

        # Telhado da restaurante
        pygame.draw.polygon(canvas, GREEN, [(mapped_x, mapped_y - house_size),
                                            (mapped_x - house_size // 2, mapped_y - house_size // 2),
                                            (mapped_x + house_size // 2, mapped_y - house_size // 2)])
        
        # Desenha o ID do estabelecimento
        id_text = self.font_id.render(str(establishment_id), True, WHITE)
        id_rect = id_text.get_rect(center=(mapped_x, mapped_y - 2))
        canvas.blit(id_text, id_rect)

    def draw_customer(self, canvas, mapped_x, mapped_y):
        # Altura total do cliente (pino + bolinha)
        pin_length = 7
        circle_radius = 5

        # A extremidade inferior do pino estará exatamente na coordenada do cliente
        pin_start_x = mapped_x
        pin_start_y = mapped_y
        pin_end_x = mapped_x
        pin_end_y = mapped_y - pin_length

        # A bolinha estará na extremidade superior do pino
        circle_center_x = pin_end_x
        circle_center_y = pin_end_y - circle_radius

        # Desenhar o pino
        pygame.draw.line(canvas, BLUE, (pin_start_x, pin_start_y), (pin_end_x, pin_end_y), 2)

        # Desenhar a bolinha
        pygame.draw.circle(canvas, BLUE, (int(circle_center_x), int(circle_center_y)), circle_radius)

    def draw_drivers_panel(self, canvas, environment):
        """Desenha painel com informações dos motoristas"""
        panel_x = self.frame_padding + self.map_area_width + self.frame_padding
        panel_y = self.frame_padding
        panel_width = self.side_panel_width
        panel_height = (self.map_area_height // 2) - self.frame_padding
        
        # Fundo do painel
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(canvas, LIGHT_BLUE, panel_rect)
        pygame.draw.rect(canvas, DARK_GRAY, panel_rect, 2)
        
        # Título
        title_text = self.font_info.render("Drivers", True, DARK_GRAY)
        canvas.blit(title_text, (panel_x + 10, panel_y + 10))
        
        # Linha horizontal
        pygame.draw.line(canvas, DARK_GRAY, 
                        (panel_x + 5, panel_y + 35), 
                        (panel_x + panel_width - 5, panel_y + 35), 1)
        
        # Lista de motoristas com scroll
        y_offset = panel_y + 45
        line_height = 35
        max_visible = (panel_height - 50) // line_height
        
        drivers = environment.state.drivers[:max_visible]
        
        for driver in drivers:
            if y_offset + line_height > panel_y + panel_height - 5:
                break
                
            # ID do motorista
            id_text = self.font_panel.render(f"ID: {driver.driver_id}", True, DARK_GRAY)
            canvas.blit(id_text, (panel_x + 10, y_offset))
            
            # Coordenadas
            coord_text = self.font_panel.render(
                f"Pos: ({driver.coordinate[0]:.1f}, {driver.coordinate[1]:.1f})", 
                True, DARK_GRAY
            )
            canvas.blit(coord_text, (panel_x + 10, y_offset + 15))
            
            # Cor do indicador de status
            status_color = driver.color
            pygame.draw.circle(canvas, status_color, 
                             (panel_x + panel_width - 20, y_offset + 12), 6)
            
            y_offset += line_height
    
    def draw_establishments_panel(self, canvas, environment):
        """Desenha painel com informações dos estabelecimentos"""
        panel_x = self.frame_padding + self.map_area_width + self.frame_padding
        panel_y = self.frame_padding + (self.map_area_height // 2) + self.frame_padding
        panel_width = self.side_panel_width
        panel_height = (self.map_area_height // 2) - (2 * self.frame_padding)
        
        # Fundo do painel
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(canvas, LIGHT_GREEN, panel_rect)
        pygame.draw.rect(canvas, DARK_GRAY, panel_rect, 2)
        
        # Título
        title_text = self.font_info.render("Establishments", True, DARK_GRAY)
        canvas.blit(title_text, (panel_x + 10, panel_y + 10))
        
        # Linha horizontal
        pygame.draw.line(canvas, DARK_GRAY, 
                        (panel_x + 5, panel_y + 35), 
                        (panel_x + panel_width - 5, panel_y + 35), 1)
        
        # Cache das coordenadas dos estabelecimentos
        if self.establishments_coords_cache is None:
            self.establishments_coords_cache = [
                (est.establishment_id, est.coordinate) 
                for est in environment.state.establishments
            ]
        
        # Lista de estabelecimentos
        y_offset = panel_y + 45
        line_height = 35
        max_visible = (panel_height - 50) // line_height
        
        establishments = self.establishments_coords_cache[:max_visible]
        
        for est_id, coordinate in establishments:
            if y_offset + line_height > panel_y + panel_height - 5:
                break
                
            # ID do estabelecimento
            id_text = self.font_panel.render(f"ID: {est_id}", True, DARK_GRAY)
            canvas.blit(id_text, (panel_x + 10, y_offset))
            
            # Coordenadas (fixas)
            coord_text = self.font_panel.render(
                f"Pos: ({coordinate[0]:.1f}, {coordinate[1]:.1f})", 
                True, DARK_GRAY
            )
            canvas.blit(coord_text, (panel_x + 10, y_offset + 15))
            
            # Indicador verde (sempre ativo)
            pygame.draw.circle(canvas, GREEN, 
                             (panel_x + panel_width - 20, y_offset + 12), 6)
            
            y_offset += line_height

    def draw_info_panel(self, canvas, environment):
        # Área do painel de informações
        panel_y = self.frame_padding + self.map_area_height + 20
        panel_height = self.info_panel_height - 20
        
        # Fundo do painel
        panel_rect = pygame.Rect(self.frame_padding, panel_y, self.map_area_width, panel_height)
        pygame.draw.rect(canvas, LIGHT_BLUE, panel_rect)
        pygame.draw.rect(canvas, DARK_GRAY, panel_rect, 2)
        
        # Título
        title_text = self.font_title.render("Simulation Statistics", True, DARK_GRAY)
        canvas.blit(title_text, (self.frame_padding + 20, panel_y + 10))
        
        # Linha horizontal abaixo do título
        pygame.draw.line(
            canvas, 
            DARK_GRAY, 
            (self.frame_padding + 10, panel_y + 45), 
            (self.frame_padding + self.map_area_width - 10, panel_y + 45), 
            1
        )
        
        # Coletar informações
        current_time = int(environment.now)
        orders_delivered = environment.state.orders_delivered
        
        # Contar pedidos aguardando entrega
        orders_waiting = sum(1 for customer in environment.state.customers 
                           if customer.status == CustumerStatus.WAITING_DELIVERY)
        
        # Contar motoristas por status
        drivers_available = sum(1 for driver in environment.state.drivers 
                               if driver.status == DriverStatus.AVAILABLE)
        drivers_picking_up = sum(1 for driver in environment.state.drivers 
                                if driver.status == DriverStatus.PICKING_UP)
        drivers_delivering = sum(1 for driver in environment.state.drivers 
                               if driver.status == DriverStatus.DELIVERING)
        
        # Total de estabelecimentos
        total_establishments = len(environment.state.establishments)
        total_drivers = len(environment.state.drivers)
        
        # Primeira coluna de informações
        col1_x = self.frame_padding + 30
        info_y = panel_y + 60
        line_spacing = 25
        
        info_texts = [
            f"Time: {current_time}",
            f"Orders Delivered: {orders_delivered}",
            f"Orders Waiting: {orders_waiting}",
        ]
        
        for i, text in enumerate(info_texts):
            rendered_text = self.font_info.render(text, True, DARK_GRAY)
            canvas.blit(rendered_text, (col1_x, info_y + i * line_spacing))
        
        # Segunda coluna de informações
        col2_x = self.frame_padding + self.map_area_width // 2 + 30
        
        info_texts_col2 = [
            f"Establishments: {total_establishments}",
            f"Drivers Available: {drivers_available}/{total_drivers}",
            f"Picking Up: {drivers_picking_up}  |  Delivering: {drivers_delivering}",
        ]
        
        for i, text in enumerate(info_texts_col2):
            rendered_text = self.font_info.render(text, True, DARK_GRAY)
            canvas.blit(rendered_text, (col2_x, info_y + i * line_spacing))

    def render(self, environment):
        self.quited = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quited = True

        if self.quited:
            return

        canvas = pygame.Surface(self.window_size)
        canvas.fill(WHITE)
        
        # Desenha a moldura
        self.draw_frame(canvas)
        
        # Desenha a grade se habilitado
        if self.draw_grid:
            self.draw_background_grid(canvas)
            self.draw_grid_coordinates(canvas)

        # Desenhar os estabelecimentos
        for establishment in environment.state.establishments:
            mapped_x, mapped_y = self.coordinate(establishment.coordinate)
            self.draw_establishment(canvas, mapped_x, mapped_y, establishment.establishment_id)

            if hasattr(establishment, "operating_radius"):
                operating_radius_mapped = map_coordinate(establishment.operating_radius, 0, 100, 0, min(self.map_area_width, self.map_area_height))
                pygame.draw.circle(canvas, GREEN, (int(mapped_x), int(mapped_y)), int(operating_radius_mapped), 1)
            
        # Desenhar os clientes
        for customer in environment.state.customers:
            if customer.status == CustumerStatus.WAITING_DELIVERY:
                mapped_x, mapped_y = self.coordinate(customer.coordinate)
                self.draw_customer(canvas, mapped_x, mapped_y)

        # Desenhar os motoristas
        for driver in environment.state.drivers:
            mapped_x, mapped_y = self.coordinate(driver.coordinate)
            self.draw_driver(canvas, driver.color, mapped_x, mapped_y, driver.driver_id)

            if driver.status in [DriverStatus.PICKING_UP, DriverStatus.PICKING_UP_WAITING, DriverStatus.DELIVERING, DriverStatus.DELIVERING_WAITING]:
                target_mapped_x, target_mapped_y = self.coordinate(driver.current_route_segment.coordinate)
                pygame.draw.line(canvas, RED, (mapped_x, mapped_y), (target_mapped_x, target_mapped_y), 2)

        # Desenhar painéis laterais
        self.draw_drivers_panel(canvas, environment)
        self.draw_establishments_panel(canvas, environment)
        
        # Desenhar painel de informações inferior
        self.draw_info_panel(canvas, environment)

        self.screen.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.fps)

    def quit(self):
        pygame.display.quit()
        pygame.quit()