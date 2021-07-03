from matplotlib import pyplot as plt
import csv

losses = []

with open("build/src/output.csv") as f:
  reader = csv.reader(f)

  for row in reader:
    losses.append(float(row[0]))

plt.plot(losses)
plt.show()