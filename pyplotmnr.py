import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 13})

x_series = np.array(['','None', 'Quad Only', 'Small', 'Mid', 'Large',''])
x_ticks = np.array([0,1,2,3,4,5,6])
y_ranked_des = np.array([-1, 0.7698, 0.9118, 0.8993, 0.9298, 0.8066, -1])
y_unranked_des = np.array([-1, 0.6842, 0.7623, 0.8280, 0.9230, 0.8066, -1])
y_mean_des = np.array([-1, 0.7270, 0.83705, 0.86365, 0.9264, 0.8066, -1])

fig = plt.figure(figsize=(7,6))
plt.xticks(x_ticks, x_series)
plt.plot(x_ticks, y_ranked_des, "o", label='ranked', color='cyan')
plt.plot(x_ticks, y_unranked_des, "o", label='unranked', color='red')
plt.plot(x_ticks, y_mean_des, "s", label='mean', color='black')
plt.gca().set_ylim([0.6,1.0])
yticks = plt.gca().yaxis.get_major_ticks()
yticks[-1].set_visible(False)
legend = plt.legend(loc='lower right')
plt.ylabel('Discriminability')
plt.xlabel('Nuisance Correction Method in Pipeline')
plt.title('Comparison of Different Processing Pipelines for BNU1 Dataset')

plt.savefig('desikan.png')
plt.clf()
