import pygame
import os
from PIL import Image
import random
import math

pygame.init()

WIN_WIDTH = 800
WIN_HEIGHT = 600
WIN_DIMENSIONS = (WIN_WIDTH, WIN_HEIGHT)

COLOR_YELLOW = (255, 255, 0)
COLOR_RED = (255, 0, 0)
COLOR_ORANGE = (255, 99, 71)

img = Image.open(os.path.join("car.png"))
img.thumbnail((80, 80), Image.ANTIALIAS)

MAIN_SURFACE = pygame.display.set_mode(WIN_DIMENSIONS)
pygame.display.set_caption("Self Driving Car")

CAR_PLAYER = pygame.image.fromstring(img.tobytes(), img.size, img.mode)
CAR_TRAFFIC = pygame.transform.flip(CAR_PLAYER, True, False)

SPRITE_DIMENSIONS = CAR_PLAYER.get_size()

LINE_LIST = [60, 140, 220, 300, 380, 460, 540]
rads = math.radians

class Player:
    def __init__(self, image, x, y, r=25, sens_dist=25, quantity=9):
        self.img = image
        self.img_dimensions = image.get_size()
        pt = self.get_blit_pt(x, y)
        self.rect = self.img.get_rect()
        self.x = self.rect[0] = pt[0]
        self.y = self.rect[1] = pt[1]
        self.intended_y = pt[1]
        self.isMoving = False
        self.angles = [0, 22.5, -22.5, 45, -45, 67.5, -67.5, 90, -90, 112.5, -112.5, 135, -135]
        self.sensor_info = []

        for angle in self.angles:
            self.sensor_info.append({
                's': (self.x + (math.cos(rads(angle)) * r), self.y + (math.sin(rads(angle)) * r)),
                'dx': math.cos(rads(angle)) * sens_dist,
                'dy': math.sin(rads(angle)) * sens_dist
            })

        self.sensors = []

        for sens_det in self.sensor_info:
            sensors = []
            s = sens_det['s']
            dx = sens_det['dx']
            dy = sens_det['dy']

            for i in range(1, quantity + 1):
                sensors.append(
                    pygame.draw.circle(
                        MAIN_SURFACE,
                        (255, 255, 255),
                        (round(s[0] + (i * dx)), round(s[1] + (i * dy))),
                        2,
                    )
                )

            self.sensors.append(sensors)

    def get_blit_pt(self, x, y):
        return (x - (self.img_dimensions[0] / 2), y - (self.img_dimensions[1] / 2))

    def move(self, direction):
        if direction == "UP":
            self.intended_y -= 80
        elif direction == "DOWN":
            self.intended_y += 80

    def shift_sensors(self, dx, dy):
        for sensor_array in self.sensors:
            for sensor in sensor_array:
                sensor[0] += dx
                sensor[1] += dy
    
    def render(self, win, sens_data, sens_vis):
        if self.y < self.intended_y:
            self.y += 4
            self.shift_sensors(0, 4)
            self.rect[1] += 4
            self.isMoving = True
        elif self.y > self.intended_y:
            self.y -= 4
            self.shift_sensors(0, -4)
            self.rect[1] -= 4
            self.isMoving = True
        else:
            self.isMoving = False

        if sens_vis:
            for ind, sensor_array in enumerate(self.sensors):
                COLOR = (255, 255, 255)

                if 1 <= sens_data[ind] <= 3:
                    COLOR = COLOR_RED
                elif 4 <= sens_data[ind] <= 6:
                    COLOR = COLOR_ORANGE
                elif 7 <= sens_data[ind] <= 9:
                    COLOR = COLOR_YELLOW

                for sensor in sensor_array:
                    pygame.draw.rect(MAIN_SURFACE, COLOR, sensor)


        win.blit(self.img, self.get_blit_pt(self.x, self.y))


class TrafficCar:
    def __init__(self, img, x, y):
        self.img = img
        self.img_dimensions = img.get_size()
        pt = self.get_blit_pt(x, y)
        self.x = pt[0]
        self.y = pt[1]
        self.rect = self.img.get_rect()
        self.rect[0] += self.x
        self.rect[1] += self.y

    def get_blit_pt(self, x, y):
        return (x - (self.img_dimensions[0] / 2), y - (self.img_dimensions[1] / 2))

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

        self.rect[0] += dx
        self.rect[1] += dy

    def render(self, win):
        win.blit(self.img, self.get_blit_pt(self.x, self.y))


class Traffic:

    drawn = 0
    trashed = 0

    def __init__(self, density, x, lane_ys):
        self.density = density
        self.cars = []
        self.lane_ys = lane_ys
        self.start = x

    def create_car(self):
        t_car = TrafficCar(CAR_TRAFFIC, self.start, random.choice(self.lane_ys))

        for car in self.cars:
            if car.rect.colliderect(t_car.rect):
                self.trashed += 1
                return False
        
        self.drawn += 1
        return t_car

    def reset(self):
        self.cars = []

    def render(self, win):

        if random.random() <= self.density:
            new_car = self.create_car()
            if new_car != False:
                self.cars.append(new_car)

        for car in self.cars:
            car.move(-5, 0)

            if car.x < 0:
                self.cars.remove(car)
            else:
                car.render(win)

        


def draw_game(player, traffic, sens_data, sens_visible):

    MAIN_SURFACE.fill((50, 50, 50))

    for y in LINE_LIST:
        pygame.draw.line(
            MAIN_SURFACE,
            (255, 255, 255),
            (0, y),
            (WIN_WIDTH, y)
        )
    
    player.render(MAIN_SURFACE, sens_data, sens_visible)
    traffic.render(MAIN_SURFACE)
    pygame.display.update()

def check_player_collision(player, traffic):
    for car in traffic.cars:
        if car.rect.colliderect(player.rect):
            return True
    
    return False

def sensor_collisions(player, traffic):
    sensor_detection = [0 for _ in player.sensors]

    for ind, sensor in enumerate(player.sensors):
        dist = 999

        for car in traffic.cars:
            for p_ind, point in enumerate(sensor):
                if (car.rect.contains(point) or car.rect.colliderect(point)) and p_ind + 1 < dist:
                    dist = p_ind + 1

        if dist != 999:
            sensor_detection[ind] = dist

    return sensor_detection


def main_loop():

    sim_quit = False
    player = Player(CAR_PLAYER, 300, 280, 25)
    p_lane = 3
    clock = pygame.time.Clock()
    traffic = Traffic(0.02, 800, [120, 200, 280, 360, 440, 520])
    sens_visible = False

    while not sim_quit:

        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sim_quit = True

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and p_lane > 1 and not player.isMoving:
                    player.move("UP")
                    p_lane -= 1
                elif event.key == pygame.K_DOWN and p_lane < 6 and not player.isMoving:
                    player.move("DOWN")
                    p_lane += 1
                elif event.key == pygame.K_s:
                    sens_visible = not sens_visible

            elif event.type == pygame.MOUSEBUTTONDOWN:
                print(pygame.mouse.get_pos())

        if check_player_collision(player, traffic):
            traffic.reset()

        detected_sensors = sensor_collisions(player, traffic)
        
        draw_game(player, traffic, detected_sensors, sens_visible)

    print(traffic.drawn, traffic.trashed)
    pygame.quit()
    quit()


main_loop()