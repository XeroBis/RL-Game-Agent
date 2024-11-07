import numpy as np
import random
import torch
import torch.optim as optim
from agents.base_agent import BaseAgent
from models.dqn_model import DQNModel
from config.dqn_config import DQN_CONFIG


class DQNAgent(BaseAgent):
    def __init__(self, env):
        state, _ = env.reset()
        n_observations = np.prod(state.shape)
        super().__init__(env)
        self.model = DQNModel(n_observations, env.action_space.n)
        self.memory = []
        self.epsilon = DQN_CONFIG["epsilon_start"]
        self.gamma = DQN_CONFIG["gamma"]
        self.batch_size = DQN_CONFIG["batch_size"]
        self.memory_size = DQN_CONFIG["memory_size"]
        self.total_reward = 0
        self.optimizer = optim.AdamW(self.model.parameters(), lr=DQN_CONFIG["learning_rate"], amsgrad=True)
    
    def choose_action(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return self.model.predict(state_tensor)
    
    def train(self, nb_episodes=300):
        for _ in range(nb_episodes):
            state, _ = self.env.reset()
            done = False
            number_step = 0
            while not done:
                action = self.choose_action(state, self.epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                self.memory.append((state, action, reward, next_state, done))
                if len(self.memory) > self.memory_size:
                    self.memory.pop(0)

                self.update_model()
                state = next_state
                self.total_reward += reward
                if number_step > 200:
                    done = True
            
            self.epsilon = max(DQN_CONFIG["epsilon_min"], self.epsilon * DQN_CONFIG["epsilon_decay"])
    
    def update_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
        
        targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = torch.nn.functional.mse_loss(current_q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def evaluate(self, num_episodes=100):
        total_rewards = 0
        rewards = []
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.choose_action(state, epsilon=0)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = next_state

            rewards.append(episode_reward)
            total_rewards += episode_reward

        average_reward = total_rewards / num_episodes
        print("MODEL :")
        print(f"Average reward over {num_episodes} episodes: {average_reward}, min : ", min(rewards), 
              ", max : ", max(rewards), ", 25 percentile : ", np.percentile(rewards, 25),
              " 75 percentile : ", np.percentile(rewards, 75))

        # Random
        total_rewards = 0
        rewards = []
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.choose_action(state, epsilon=2)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = next_state

            rewards.append(episode_reward)
            total_rewards += episode_reward

        average_reward = total_rewards / num_episodes
        print("RANDOM :")
        print(f"Average reward over {num_episodes} episodes: {average_reward}, min : ", min(rewards), 
              ", max : ", max(rewards), ", 25 percentile : ", np.percentile(rewards, 25),
              " 75 percentile : ", np.percentile(rewards, 75))
        
