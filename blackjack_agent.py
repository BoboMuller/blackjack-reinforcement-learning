import random
from collections import defaultdict
import numpy as np


class BlackjackAgent:
    def __init__(self):
        self.actions = np.array([0, 1], dtype=int)
        self.memory = defaultdict(lambda: [0, 0])
        self.epsilon = 1.0
        self.decay = 0.9999
        self.learning_rate = 0.01
        self.discount_factor = 0.7

    def action(self, observation, exploit_only=False) -> int:
        if not exploit_only and observation[0] < 11:
            # We have a guarantee that you won't be busted with a hand of 10
            ret = 1
        elif not exploit_only and random.random() < self.epsilon:
            ret = random.choice(self.actions)
        else:
            ret = np.argmax(self.memory[observation])
        self.epsilon = max(0.1, self.epsilon * self.decay) # Reduce the randomness at each step
        return ret

    def learn(
        self, action, observation, reward, terminated, next_observation
    ) -> float:
        """Returns the current error"""
        predict = self.memory[observation][action]
        if not terminated:
            target = reward + self.discount_factor * max(self.memory[next_observation])
        else:
            target = reward
        error = target - predict
        self.memory[observation][action] += self.learning_rate * error
        return error
