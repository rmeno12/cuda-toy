from matplotlib import pyplot as plt
import csv

losses = []
times = []

with open("build/src/output.csv") as f:
  reader = csv.reader(f)

  for row in reader:
    losses.append(float(row[0]))
    times.append(int(row[1]))

plt.rcParams.update({'font.size': 22})
fig, axs = plt.subplots(1, 2)

axs[0].plot(losses)
axs[0].set_xlabel('Iterations')
axs[0].set_ylabel('Batch Average Loss')
axs[0].set_title('Loss over time')

axs[1].hist(times, bins=30)
axs[1].set_xlabel('Time per iteration (microseconds)')
axs[1].set_ylabel('Count')
axs[1].set_title('Distribution of Iteration Times')
plt.show()