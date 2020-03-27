# The amount by which the agent shifts each frame if moving.
# Note: Should be a factor of 80
player_shift_magnitude = 40

# The distance of the start of the sensor arms from the center of the agent.
sensor_radius = 35

# The distance between the sensor dots on a sensor arm.
sensor_dot_distance = 35

# The frequency at which cars appear. This should ideally be a low value.
traffic_density = 0.02

# Sensor arms colors when rendering.
sensor_col_detecting_nothing = (255, 255, 255)
sensor_col_detecting_far = (255, 255, 0)
sensor_col_detecting_nearer = (255, 99, 71)
sensor_col_detecting_nearest = (255, 0, 0)

# Agent rewards
# Hit Reward:   When the agent hits a car.
# Dodge Reward: When the agent successfully dodges a car. The agent will continuously get
#               this amount of rewards as long as it remains on the side of a traffic car.
# Ide Reward:   The reward the agent gets for doing nothing.
hit_reward = -500
dodge_reward = 50
idle_reward = 1

# NEURAL NETWORK HYPERPARAMETERS
# Number of neurons in the hidden layers.
neurons_in_layer_1 = 256
neurons_in_layer_2 = 256

# Neural network activation functions.
# Note: should be a valid string representation of a built-in Tensorflow Activation Function.
nn_activation_layer_1 = 'relu'
nn_activation_layer_2 = 'relu'

# Adam optimizer learning rate.
optimizer_learning_rate = 1e-3

# Loss Function
# Note: should be a valid string representation of a built-in Tensorflow Loss Function.
model_loss_function = 'mse'

# TRAINING HYPERPARAMETERS
# Replay buffer max size
replay_buffer_size = 1000000

# Batch size
training_batch_size = 100

# Target Q value discount factor
target_Q_discount = 0.95

# Starting epsilon
# Note: epsilon is always 0.01 during retraining
epsilon = 1

# Epsilon decrement factor. The epsilon is multiplied by this factor at every step.
epsilon_decrement_factor = 0.999

# Number of episodes you want the model to train for.
# Note that you can stop training at any time by quitting the pygame window after which
# the code will save your model and replay buffer and you won't lose anything.
number_of_episodes = 100000

# The amount of steps after which the target model adopts the weights
# of the Evaluator model.
switch_weights_step_interval = 100

# FILENAMES
# The filename with which your model is saved.
# Note: needs to have an h5 extension
model_fname = 'model.h5'

# Filename of the replay buffer pickle.
# Note: pickle extension
replay_buffer_fname = 'replay_buffer.pickle'