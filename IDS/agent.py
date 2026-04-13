from dataSet import X_seq
from collections import deque
import keras 
from keras.models import Sequential 
from keras.layers import Input, LSTM, Dense
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
        Input(shape=(10, X_seq.shape[2])), # 10 timesteps, 118 features
        LSTM(64, return_sequences=False), # Number of LSTM units
        Dense(32, activation='relu'), 
        Dense(2, activation='linear') # Q-values for action 0 and action 1
    ])
    return model

main_network = build_q_network()
main_network.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse'
)
target_network = build_q_network()
target_network.set_weights(main_network.get_weights())

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
        # dont know exactly what this below 2 lines codes  
        state_batch = np.expand_dims(state, axis=0)
        # I think this line takes the q_values that is predicted by the main network
        q_values = main_network.predict(state_batch, verbose=0) # type: ignore

        # pick the action (index) with the highest Q_value
        return np.argmax(q_values[0])

# not understood function this part
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
            target = rewards[i] + gamma * np.amax(next_q_values[i])
            
        current_q_values[i][actions[i]] = target

    # Train the network on the updated values
    main_network.fit(states, current_q_values, batch_size=batch_size, epochs=1, verbose=0) # type: ignore


def update_target_network():
    target_network.set_weights(main_network.get_weights())
