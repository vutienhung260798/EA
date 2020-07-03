import numpy as np
from environments import CartPoleEnv
from environments import AcrobotEnv

class Sphere:

    def __init__(self, dim):
        self.dim = dim

    def fitness(self, x):
        return np.sum(np.power(x, 2))
    

class CartPole:
    
    def __init__(self, gravity):
        self.dim = 5
        self.env = CartPoleEnv()
        self.env.gravity = gravity

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def action(self, observation, x):
        x = x * 10 - 5
        w = x[:4]
        b = x[4]
        return int(self.sigmoid(np.sum(observation * w) + b) > 0.5)

    def fitness(self, x):
        fitness = 0
        observation = self.env.reset()
        for t in range(200):
            action = self.action(observation, x)
            observation, reward, done, info = self.env.step(action)
            fitness += reward
            if done:
                break
        return - fitness

    def __del__(self):
        self.env.close()

    
class Acrobot:
    
    def __init__(self, link_mass_2):
        self.dim = 7
        self.env = AcrobotEnv()
        self.env.LINK_MASS_2 = link_mass_2

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def action(self, observation, x):
        x = x * 10 - 5
        w = x[:6]
        b = x[6]
        return int(self.sigmoid(np.sum(observation * w) + b) > 0.5)

    def fitness(self, x):
        fitness = 0
        observation = self.env.reset()
        for t in range(200):
            action = self.action(observation, x)
            observation, reward, done, info = self.env.step(action)
            fitness += reward
            if done:
                break
        return - fitness

    def __del__(self):
        self.env.close()

def main():
    task = CartPole(9.8)
    y = task.fitness(np.random.rand(5))

    print(y)

if __name__ == '__main__':
    main()        

