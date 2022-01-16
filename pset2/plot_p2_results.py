import matplotlib.pyplot as plt

with open('p2_results.txt', 'r') as f:
    lines = f.read().split('\n')

takeoff_time = 0
takeoff_pos = 0
pos = []
vel = []
thrust = []
braking = []

takeoff_time, takeoff_pos, pos, vel, thrust, braking, _ = lines

takeoff_pos = float(takeoff_pos)
takeoff_time = int(takeoff_time)
pos = list(map(lambda x: float(x), pos.split(', ')))
vel = list(map(lambda x: float(x), vel.split(', ')))
thrust = list(map(lambda x: float(x), thrust.split(', ')))
braking = list(map(lambda x: float(x), braking.split(', ')))

x = range(takeoff_time + 1)

plt.plot(x, vel, label = 'velocity')
plt.plot(x, thrust, label = 'thrust')
plt.plot(x, braking, label = 'braking')

plt.xlabel('time')
plt.title('Min Time Takeoff')
plt.legend()

plt.savefig('p2_plot_no_position.png')

plt.plot(x, pos, label = 'position')

plt.legend()
plt.savefig('p2_plot.png')
