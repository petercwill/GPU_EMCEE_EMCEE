import matplotlib.pylab as plt
import numpy as np

file1 = "out.txt"
file2 = "out2.txt"

xs1 = []
ys1 = []

xs2 = []
ys2 = []

x_means = []
y_means = []


with open(file1, "r") as f1:
    for line in f1.readlines():
        items = line.split("\t")
        xs1.append(items[0])
        ys1.append(items[1])

with open(file2, "r") as f2:
    for line in f2.readlines():
        items = line.split("\t")
        xs2.append(items[0])
        ys2.append(items[1])

with open("means.txt", "r") as f3:
    for line in f3.readlines():
        items = line.split("\t")
        x_means.append(items[0])
        y_means.append(items[1])

fig, ax = plt.subplots(1,1)

ax.scatter(xs1, ys1)
ax.scatter(xs2, ys2, c='r')
plt.show()

ts = [i for i in range(len(x_means))]
plt.plot(ts, x_means)
plt.show()
plt.plot(ts, y_means)
plt.show()
