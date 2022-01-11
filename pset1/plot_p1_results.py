import matplotlib.pyplot as plt

with open('p1_results.txt', 'r') as f:
    lines = f.read().split('\n')

lows = []
highs = []
for line in lines[:-1]:
    low, high = line.split(' ')
    lows.append(float(low))
    highs.append(float(high))

x = range(120)

plt.plot(x, lows)
plt.plot(x, highs)

plt.xlabel('months')
plt.ylabel('discounting')

plt.savefig('p1_plot.png')
