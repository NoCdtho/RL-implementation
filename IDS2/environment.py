class IDSEnvironment:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self.data[self.current_step]

    def step(self, action):
        true_label = self.labels[self.current_step]
        
        # Reward Logic
        if action == true_label:
            reward = 2.0 if true_label == 1 else 1.0 # High reward for catching attacks
        else:
            reward = -5.0 if true_label == 1 else -1.0 # Heavy penalty for False Negatives

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        next_state = self.data[self.current_step] if not done else None
        
        return next_state, reward, done, true_label