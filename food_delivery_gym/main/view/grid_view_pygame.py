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

def map_coordinate(value, min_val, max_val, min_screen, max_screen):
    return min_screen + (value - min_val) * (max_screen - min_screen) / (max_val - min_val)


class GridViewPygame(FoodDeliveryView):

    def __init__(self, grid_size=100, draw_grid=True, window_size=(900, 700), fps=30):
        super().__init__(grid_size, window_size, fps)
        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode(window_size)
        self.clock = pygame.time.Clock()
        self.draw_grid = draw_grid
        pygame.display.set_caption('Map of Establishments, Customers and Drivers')

    def coordinate(self, coordinate):
        return (map_coordinate(coordinate[0], self.min_x, self.max_x, 0, self.window_size[0]),
                map_coordinate(coordinate[1], self.min_y, self.max_y, 0, self.window_size[1]))
    
    def draw_background_grid(self, canvas, color=GRAY):
        # Desenha a grade
        grid_size_x = self.window_size[0] // self.grid_size
        grid_size_y = self.window_size[1] // self.grid_size

        for x in range(0, self.window_size[0], grid_size_x):
            pygame.draw.line(canvas, color, (x, 0), (x, self.window_size[1]), 1)
        for y in range(0, self.window_size[1], grid_size_y):
            pygame.draw.line(canvas, color, (0, y), (self.window_size[0], y), 1)
    
    def draw_driver(self, canvas, driver_color, mapped_x, mapped_y):
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


    def draw_establishment(self, canvas, mapped_x, mapped_y):
        # Corpo do restaurante
        house_size = 15
        pygame.draw.rect(canvas, GREEN, (mapped_x - house_size // 2, mapped_y - house_size // 2, house_size, house_size))

        # Telhado da restaurante
        pygame.draw.polygon(canvas, GREEN, [(mapped_x, mapped_y - house_size),
                                            (mapped_x - house_size // 2, mapped_y - house_size // 2),
                                            (mapped_x + house_size // 2, mapped_y - house_size // 2)])

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


    def render(self, environment):
        self.quited = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quited = True

        if self.quited:
            return

        canvas = pygame.Surface(self.window_size)

        canvas.fill(WHITE)
        
        if self.draw_grid:
            self.draw_background_grid(canvas)

        # Desenhar os estabelecimentos
        for establishment in environment.state.establishments:
            mapped_x, mapped_y = self.coordinate(establishment.coordinate)
            self.draw_establishment(canvas, mapped_x, mapped_y)

            if hasattr(establishment, "operating_radius"):
                operating_radius_mapped = map_coordinate(establishment.operating_radius, 0, 100, 0, min(self.window_size))
                pygame.draw.circle(canvas, GREEN, (int(mapped_x), int(mapped_y)), int(operating_radius_mapped), 1)
            
        # Desenhar os clientes
        for customer in environment.state.customers:
            if customer.status == CustumerStatus.WAITING_DELIVERY:
                mapped_x, mapped_y = self.coordinate(customer.coordinate)
                self.draw_customer(canvas, mapped_x, mapped_y)

        # Desenhar os motoristas
        for driver in environment.state.drivers:
            mapped_x, mapped_y = self.coordinate(driver.coordinate)
            self.draw_driver(canvas, driver.color, mapped_x, mapped_y)

            if driver.status in [DriverStatus.PICKING_UP, DriverStatus.DELIVERING]:
                target_mapped_x, target_mapped_y = self.coordinate(driver.current_route_segment.coordinate)
                pygame.draw.line(canvas, RED, (mapped_x, mapped_y), (target_mapped_x, target_mapped_y), 2)

        self.screen.blit(canvas, canvas.get_rect())
        pygame.display.update()
        self.clock.tick(self.fps)

    def quit(self):
        pygame.display.quit()
        pygame.quit()
