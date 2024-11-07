class BaseAgent:
    def __init__(self, env):
        self.env = env
        self.total_reward = 0

    def choose_action(self, state):
        raise NotImplementedError("This method should be overridden in subclasses.")
    
    def train(self):
        raise NotImplementedError("This method should be overridden in subclasses.")

    def evaluate(self):
        raise NotImplementedError("This method should be overridden in subclasses.")
