import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('/home/hung-vt/MFEA/data/acrobot_cartpole/mt_11.csv')
# print(data.columns)

x = data['0']
y1 = - data['191.150000']
y2 = - data['-18.550000']

def plot(x, y, label):
    plt.plot(x, y, label = label)

plot(x, y1, 'acrobot')
plt.ylabel('Fitness')
plt.xlabel('Generations')
plt.legend()
plt.show()
plot(x, y2, 'cartpole')
# plt.title(')
plt.ylabel('Fitness')
plt.xlabel('Generations')
plt.legend()
plt.show()
# print(y2)