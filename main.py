import pygame
import os
from PIL import Image
import random
import math
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import datetime
import pickle
import consts_and_hyperparams as cs

pygame.init()

WIN_WIDTH = 800
WIN_HEIGHT = 800
WIN_DIMENSIONS = (WIN_WIDTH, WIN_HEIGHT)

img = Image.open(os.path.join("car.png"))
img.thumbnail((80, 80), Image.ANTIALIAS)
CAR_PLAYER = pygame.image.fromstring(img.tobytes(), img.size, img.mode)

img = Image.open(os.path.join('car2.png'))
img.thumbnail((80, 80), Image.ANTIALIAS)
img = img.rotate(90, expand=True)
CAR_TRAFFIC = pygame.image.fromstring(img.tobytes(), img.size, img.mode)

MAIN_SURFACE = pygame.display.set_mode(WIN_DIMENSIONS)
pygame.display.set_caption("Self Driving Car")

LINE_LIST = [160, 240, 320, 400, 480, 560, 640]
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
        self.is_moving = False
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

        shift_qty = cs.player_shift_magnitude

        
        if self.y < self.intended_y:
            self.y += shift_qty
            self.shift_sensors(0, shift_qty)
            self.rect[1] += shift_qty
            self.is_moving = True
        elif self.y > self.intended_y:
            self.y -= shift_qty
            self.shift_sensors(0, -shift_qty)
            self.rect[1] -= shift_qty
            self.is_moving = True
        else:
            self.is_moving = False
        

    def shift_sensors(self, dx, dy):
        for sensor_array in self.sensors:
            for sensor in sensor_array:
                sensor[0] += dx
                sensor[1] += dy
    
    def render(self, win, sens_data, sens_vis):

        if sens_vis:
            for ind, sensor_array in enumerate(self.sensors):
                COLOR = cs.sensor_col_detecting_nothing

                if 1 <= sens_data[ind] <= 3:
                    COLOR = cs.sensor_col_detecting_nearest
                elif 4 <= sens_data[ind] <= 6:
                    COLOR = cs.sensor_col_detecting_nearer
                elif 7 <= sens_data[ind] <= 9:
                    COLOR = cs.sensor_col_detecting_far

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
        self.reward_rects = [
            pygame.Rect(self.x, self.y + 80, 80, 40),
            pygame.Rect(self.x, self.y - 80, 80, 40)
        ]

    def get_blit_pt(self, x, y):
        return (x - (self.img_dimensions[0] / 2), y - (self.img_dimensions[1] / 2))

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

        self.rect[0] += dx
        self.rect[1] += dy

        for reward_rect in self.reward_rects:
            reward_rect[0] += dx
            reward_rect[1] += dy

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

    def move_cars(self):
        if random.random() <= self.density:
            new_car = self.create_car()
            if new_car != False:
                self.cars.append(new_car)

        for car in self.cars:
            car.move(-5, 0)

            if car.x < 0:
                self.cars.remove(car)

    def render(self, win):
        self.move_cars()
        for car in self.cars:
            car.render(win)


def draw_game(player, traffic, sens_data, sens_visible=True):

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

def check_dodge(player, traffic):
    for car in traffic.cars:
        for reward_rect in car.reward_rects:
            if player.rect.colliderect(reward_rect):
                return cs.dodge_reward

    return cs.idle_reward

def sensor_collisions(player, traffic, print_sens_data=False):
    sensor_detection = [0 for _ in player.sensors]

    for ind, sensor in enumerate(player.sensors):
        dist = 999

        for car in traffic.cars:
            for p_ind, point in enumerate(sensor):
                if (car.rect.contains(point) or car.rect.colliderect(point)) and p_ind + 1 < dist:
                    dist = p_ind + 1

        if dist != 999:
            sensor_detection[ind] = dist

    if print_sens_data:
        print(sensor_detection)

    return sensor_detection


class Env:
    sim_quit = False
    player = Player(CAR_PLAYER, 300, 380, cs.sensor_radius, cs.sensor_dot_distance)
    p_lane = 3
    clock = pygame.time.Clock()
    traffic = Traffic(cs.traffic_density, 800, [220, 300, 380, 460, 540, 620])
    sens_visible = True
    print_sens_data = False

    def step(self, action):
        if action == 1 and self.p_lane > 1 and not self.player.is_moving:
            self.player.move('UP')
            self.p_lane -= 1
        if action == 2 and self.p_lane < 6 and not self.player.is_moving:
            self.player.move('DOWN')
            self.p_lane += 1
        if action == 0:
            self.player.move('STAY')

        self.traffic.move_cars()

        sensor_data = sensor_collisions(self.player, self.traffic)
        next_state = [s for s in sensor_data]
        next_state.append(self.p_lane)
        next_state.append(int(self.player.is_moving))

        done = check_player_collision(self.player, self.traffic)

        reward = cs.hit_reward if done else check_dodge(self.player, self.traffic)

        return [next_state, reward, done]

    def reset(self):
        self.traffic.reset()
        arr = [0 for _ in range(13)]
        arr.append(0)
        arr.append(3)
        return arr

    def render(self, win, sens_data=[], show_sensors=False):
        draw_game(self.player, self.traffic, sens_data, show_sensors)

n_outputs = 3

model = keras.models.Sequential([
    keras.layers.Dense(cs.neurons_in_layer_1, input_shape=(15,)),
    keras.layers.Activation(cs.nn_activation_layer_1),
    keras.layers.Dense(cs.neurons_in_layer_2),
    keras.layers.Activation(cs.nn_activation_layer_2),
    keras.layers.Dense(n_outputs)
])

target = keras.models.clone_model(model)
target.set_weights(model.get_weights())

optimizer = keras.optimizers.Adam(learning_rate=cs.optimizer_learning_rate)

model.compile(optimizer=optimizer, loss=cs.model_loss_function)
target.compile(optimizer=optimizer, loss=cs.model_loss_function)

replay_buffer = deque(maxlen=cs.replay_buffer_size)

def epsilon_greedy_policy(state, epsilon=0):
    action = 0
    if np.random.rand() < epsilon:
        action = np.random.randint(3)
    else:
        Q_values = model.predict(np.array(state)[np.newaxis])
        action = np.argmax(Q_values[0])

    return action

def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch]) 
        for field_index in range(5)
    ]

    return states, actions, rewards, next_states, dones

def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done = env.step(action)
    replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done


batch_size = cs.training_batch_size
discount_factor = cs.target_Q_discount

def training_step(batch_size):
    states, actions, rewards, next_states, dones = sample_experiences(batch_size)
    dones = [1 - x for x in dones]
    next_Q_values = target.predict(next_states)
    eval_Q_values = model.predict(next_states)
    max_actions = np.argmax(eval_Q_values, axis=1)
    target_Q_values = model.predict(states)

    batch_indices = np.arange(batch_size, dtype=np.int32)

    target_Q_values[batch_indices, actions] = rewards + discount_factor * \
        next_Q_values[batch_indices, max_actions.astype(int)] * dones

    model.fit(states, target_Q_values, verbose=0)

    '''
    mask = tf.one_hot(actions, n_outputs)
    all_Q_values = model(states)
    Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
    loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    '''


env = Env()
eps = []
rews = []

retraining = False
render_game = False

def load_model():
    model = keras.models.load_model(cs.model_fname)
    target = keras.models.load_model(cs.model_fname)

    with open(cs.replay_buffer_fname, 'rb') as rb_file:
        replay_buffer = pickle.load(rb_file)

def save_model():
    model.save(cs.model_fname, overwrite=True)
    with open(cs.replay_buffer_fname, 'wb') as rb_file:
        pickle.dump(replay_buffer, rb_file)

def quit_everything():
    plt.plot(eps, rews)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    #plt.savefig(f"{datetime.datetime.now()}.png")
    plt.show()
    pygame.quit()
    quit()

def train_model():
    if retraining:
        load_model()

    total_steps = 0
    epsilon = cs.epsilon
    epsilon_dec_rate = cs.epsilon_decrement_factor
    exps_stored = 0
    learn_target = cs.switch_weights_step_interval
    show_sensors = False
    for episode in range(cs.number_of_episodes):
        ep_done = False
        obs = env.reset()
        ep_rews = 0
        while not ep_done:
            obs, reward, done = play_one_step(env, obs, max(epsilon, 0.01) if not retraining else 0.01)
            exps_stored += 1
            ep_rews += reward
            ep_done = done

            if render_game:
                if not show_sensors:
                    env.render(MAIN_SURFACE)
                else:
                    env.render(MAIN_SURFACE, sensor_collisions(env.player, env.traffic), True)

            if exps_stored > batch_size:
                training_step(batch_size)

            if exps_stored % learn_target == 0:
                target.set_weights(model.get_weights())

            epsilon *= epsilon_dec_rate

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    save_model()
                    quit_everything()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        show_sensors = not show_sensors
                    if event.key == pygame.K_q:
                        save_model()
                        quit_everything()

        eps.append(episode)
        rews.append(ep_rews)
        save_model()
        print(f"Episode {episode} done. Rewards = {ep_rews}")
            
    save_model()
    quit_everything()
    
    

def get_max_index(arr):
    max_val = -999
    max_ind = -999

    for ind, val in enumerate(arr):
        if val > max_val:
            max_val = val
            max_ind = ind

    return max_ind

def action_to_string(action):
    if action == 0:
        return 'STAY'
    elif action  == 1:
        return 'UP'
    else:
        return 'DOWN'

testing = False
def main_loop():
    
    if testing:
        sim_model = keras.models.load_model('model.h5')

    sim_quit = False
    clock = pygame.time.Clock()

    obs = env.reset()
    show_sensors = False
    show_predictions = False
    while not sim_quit:

        action = 0

        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sim_quit = True

            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and not testing:
                    action = 1
                elif event.key == pygame.K_DOWN and not testing:
                    action = 2
                elif event.key == pygame.K_s:
                    show_sensors = not show_sensors
                elif event.key == pygame.K_a and testing:
                    show_predictions = not show_predictions
                elif event.key == pygame.K_q:
                    sim_quit = True
            
        if testing:
            prediction = sim_model.predict(np.array(obs)[np.newaxis])
            action = get_max_index(prediction[0])

            if show_predictions:
                print(prediction[0], action_to_string(action))

        obs, reward, done = env.step(action)

        sens_data = obs[:]
        sens_data.pop()
        sens_data.pop()

        env.render(MAIN_SURFACE, sens_data, show_sensors)

        if done:
            env.reset()

    pygame.quit()
    quit()


train_or_test = input("Train/Retrain/Test/Play? (1=Train, 2=Retrain, 3=Test, 4=Play): ")

def ask_for_render():
    render_or_not = input("Wanna Render the game? (1=Yes, 0=No): ")

    if render_or_not == '1':
        render_game = True
    elif render_or_not == '0':
        render_game = False
    else:
        pygame.quit()
        quit()

if train_or_test == '1':
    ask_for_render()
    train_model()
elif train_or_test == '2':
    retraining = True
    ask_for_render()
    train_model()
elif train_or_test == '3':
    testing = True
    main_loop()
elif train_or_test == '4':
    main_loop()
