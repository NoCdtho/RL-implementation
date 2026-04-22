from dataset import IoTDataset # Ensure your file is named dataPreprocessing.py
from agent import DQNAgent
from environment import IDSEnvironment
import matplotlib.pyplot as plt
import torch

# 1. Setup
data_manager = IoTDataset('CICIOT2023.csv')
X, y = data_manager.get_data()
env = IDSEnvironment(X, y)
agent = DQNAgent(input_dim=11)
episode_rewards = []

# 2. Training Loop
episodes = 50
batch_size = 32
step_count = 0 # Added step counter for performance

for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    
    while True:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        
        if next_state is not None:
            agent.memory.append((state, action, reward, next_state, done))
        
        state = next_state
        total_reward += reward
        
        # FIX: Only run backpropagation every 8 steps to vastly speed up training
        if step_count % 8 == 0:
            agent.learn(batch_size)
            
        # FIX: Sync Target Network during the episode (optional, but recommended for stability)
        if step_count % 1000 == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())
            
        step_count += 1
        if done: break

    # Decay Epsilon at the end of each episode
    agent.epsilon = max(0.01, agent.epsilon * 0.95)
    episode_rewards.append(total_reward)
    
    # FIX: Indented to print at the end of EVERY episode
    print(f"Episode {ep+1} | Total Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.2f}")

# 3. Generating Outputs
plt.figure(figsize=(10, 5))
plt.plot(range(1, episodes + 1), episode_rewards, marker='o', color='b')
plt.title('AI Learning Progress: Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
plt.savefig('training_rewards.png', dpi=300)
plt.show()

# 4. Save the PyTorch Model
torch.save(agent.model.state_dict(), "lstm_dqn_model.pth")
print("Saved trained model to 'lstm_dqn_model.pth'")