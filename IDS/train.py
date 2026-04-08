from agent import main_network
import matplotlib.pyplot as plt
from dataSet import X_seq, y_seq
from environment import IntrusionDetectionEnv
from agent import choose_action, remember, replay, update_target_network
from agent import epsilon, epsilon_decay, epsilon_min, batch_size

# 1. Initialize the Environment
env = IntrusionDetectionEnv(X_seq, y_seq)

# Number of complete passes through the dataset
EPISODES = 5 
current_epsilon = epsilon # Track epsilon locally during the loop

episode_rewards = []

for e in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    
    print(f"--- Starting Episode {e + 1} ---")

    while not done:
        # 2. Agent chooses an action
        action = choose_action(state, current_epsilon)
        
        # 3. Environment processes the action
        next_state, reward, done = env.step(action)
        
        # 4. Agent remembers the interaction
        remember(state, action, reward, next_state, done)
        
        # 5. Agent learns from past memories
        replay(batch_size)
        
        # 6. Decay exploration rate
        if current_epsilon > epsilon_min:
            current_epsilon *= epsilon_decay
            
        # Move forward
        state = next_state
        total_reward += reward
        step_count += 1

        
        # 7. Sync networks at the end of the episode
        if step_count % 100 == 0:
            update_target_network()
            episode_rewards.append(total_reward)
        
        # Print progress to terminal
        if step_count % 1000 == 0:
            print(f"Step: {step_count}, Current Epsilon: {current_epsilon:.4f}")


    print(f"Episode {e + 1} Finished! Total Steps: {step_count}, Total Reward: {total_reward}, Final Epsilon: {current_epsilon:.4f}\n")

plt.figure(figsize=(10, 5))
plt.plot(range(1, EPISODES + 1), episode_rewards, marker='o', color='b')
plt.title('AI Learning Progress: Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
main_network.save("lstm_dqn_model.keras")
plt.show()
