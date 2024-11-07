import argparse
from agents.dqn_agent import DQNAgent

import gym

def main(args):
    env = gym.make(args.env_name)
    agent = DQNAgent(env)
    agent.train()
    print("Agent :", "DQN")
    print("Env :", args.env_name)
    agent.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, choices=['CartPole-v1','Pong-v4', 'AirRaid-v4', 'Breakout-v4'],default="CartPole-v1")
    args = parser.parse_args()
    main(args)
