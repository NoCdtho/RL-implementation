# agent.py
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense
from keras.optimizers import Adam

class DQNAgent:
    def __init__(self, input_shape):
        # Hyperparameters
        self.epsilon = 1.0  
        self.epsilon_decay = 0.995 
        self.epsilon_min = 0.01 
        self.action_size = 2 
        self.gamma = 0.95 
        
        # Buffer
        self.memory = deque(maxlen=5000)
        
        # Networks (Using your exact architecture)
        self.main_network = self._build_q_network(input_shape)
        self.target_network = self._build_q_network(input_shape)
        self.update_target_network()

    def _build_q_network(self, input_shape):
        model = Sequential([
            Input(shape=input_shape), 
            LSTM(64, return_sequences=False),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_batch = np.expand_dims(state, axis=0)
        q_values = self.main_network.predict(state_batch, verbose=0)
        return np.argmax(q_values[0])

    def learn(self, batch_size):
        # Your exact Bellman Equation logic goes here!
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        # ... (Insert the rest of your replay logic here) ...
        # After training, you can decay epsilon here or in the main loop
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())