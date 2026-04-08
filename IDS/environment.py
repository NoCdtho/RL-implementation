import numpy as np

class IntrusionDetectionEnv:
    def __init__(self, X_seq, y_seq, false_negative_penalty=-5):
        self.X_seq = np.array(X_seq)
        self.y_seq = np.array(y_seq)

        # this tarcker variable starting at 0
        self.current_step = 0

        #Action space:
        # 0 = Predict Normal 
        # 1 = Predict Attack
        self.action_space = [0, 1]
        
        self.false_negative_penalty = false_negative_penalty

    def reset(self):
        self.current_step = 0
        return self.X_seq[0]
        
    def step(self, action):
        true_label = self.y_seq[self.current_step]

        # Reward logic
        if action == 1 and true_label == 1:
            reward = 1
        elif action == 0 and true_label == 0:
            reward = 1
        elif action == 1 and true_label == 0:
            reward = -1
        elif action == 0 and true_label == 1:
            reward = self.false_negative_penalty
        else:
            raise ValueError('Invalid action or label')
            
        self.current_step += 1

        done = self.current_step >= len(self.X_seq)

        if done:
            next_step = np.zeros_like(self.X_seq[0])
        else:
            next_step = self.X_seq[self.current_step]

        return next_step, reward, done
