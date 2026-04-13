from collections import deque
import keras
from keras.models import Sequential 
from keras.layers import LSTM, Dense, Input
from keras.optimizers import Adam
import numpy as np
import random

# Initial hyperparameter
epsilon = 1.0  # start with 100 % exploration
epsilon_decay = 0.995 # Multiply epsilon by this number every step
epsilon_min = 0.01 # never explore 1 less than 1%
action_size = 2 
gamma = 0.95 # discount factor (future reward weight)
batch_size = 32 # number of memories to train on at once

# Building the basic LSTM DQN
def build_q_network():
    model = Sequential([
        #  Input is a window of data that AI sees at once.
        Input(shape=(10, 122)),
        # Unlike a standard "Dense" layer, the LSTM has a hidden state (memory) that allows it to pass information from one timestep to the next. 
        LSTM(64), 
        # this is the standard fully connected layer
        Dense(32, activation='relu'), 
        # This is the output layer
        Dense(2, activation='linear') 
    ])
    return model

main_network = build_q_network()
main_network.compile(
    optimizer=Adam(learning_rate=0.001), # This is used to to optimize how gradients are applied.
    # Adam is the algo that applies the gradient carefully in th weight.
    # gradient is the partial derivative between loss-fucntion and weight and determines which direction to move the dial 
    loss='mse' # this is mean squared error this is the difference between what reward was predicted by ai what the actual reward was.
)
target_network = build_q_network()
target_network.set_weights(main_network.get_weights()) # This line copies the random starting weights from the main_network into the target_network.

# Adding the adaptive learning 
replay_buffer = deque(maxlen=5000)

def remember(state, action, reward, next_state, done):
    replay_buffer.append((state, action, reward, next_state, done))

def choose_action(state, epsilon):
    # Generate a random number between 0 and 1. If it's less than epsilon
    if np.random.rand() <= epsilon:
        # return a random number 0, 1
        return random.randrange(action_size)
    else:
        # keras model refuses to look at 1 item they demand a batch of items this line wraps a state into a outer and keras think's it's 1 batch of item.  
        state_batch = np.expand_dims(state, axis=0)
        # I think this line takes the q_values that is predicted by the main network
        q_values = main_network.predict(state_batch, verbose=0) # type: ignore
        # pick the highest Q_value
        return np.argmax(q_values[0])

def replay(batch_size):
    if len(replay_buffer) < batch_size:
        return
    
    minibatch = random.sample(replay_buffer, batch_size)

    # Unpack the tuple batches 
    states = np.array([transition[0] for transition in minibatch])
    actions = np.array([transition[1] for transition in minibatch])
    rewards = np.array([transition[2] for transition in minibatch])
    next_states = np.array([transition[3] for transition in minibatch])
    dones = np.array([transition[4] for transition in minibatch])
    
    # Predict Q-values
    current_q_values = main_network.predict(states, verbose=0) # type: ignore
    next_q_values = target_network.predict(next_states, verbose=0) # type: ignore

    # Apply the Bellman Equation
    for i in range(batch_size):
        if dones[i]:
            target = rewards[i]
        else:
            # this the bellman equation
            target = rewards[i] + gamma * np.amax(next_q_values[i])

        current_q_values[i][actions[i]] = target

    # Train the network on the updated values
    main_network.fit(states, current_q_values, batch_size=batch_size, epochs=1, verbose=0) # type: ignore


def update_target_network():
    target_network.set_weights(main_network.get_weights())
