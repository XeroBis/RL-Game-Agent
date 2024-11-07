import torch
import torch.nn as nn
import numpy as np

class DQNModel(nn.Module):
    def __init__(self, nb_observations, nb_actions):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(nb_observations, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, nb_actions)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)
    
    def predict(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.forward(state)
        return torch.argmax(q_values).item()