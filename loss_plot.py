import numpy as np
import matplotlib.pyplot as plt

f = open("model-saves/final-1080p-850epoch/log.csv", "r")
lines = f.readlines()

epochs = np.zeros(len(lines)-1, dtype=int)
train_loss = np.zeros(len(lines)-1, dtype=float)
valid_loss = np.zeros(len(lines)-1, dtype=float)

for i in range(1, len(lines)):
    c = lines[i].split(";")
    epochs[i-1] = int(c[0])
    train_loss[i - 1] = float(c[1])
    valid_loss[i - 1] = float(c[2].strip())

f.close()

plt.title("1080p 850e")
plt.plot(epochs, valid_loss, color='r', label="Validation loss")
plt.plot(epochs, train_loss, color="b", label="Training loss")
plt.xlabel("Number of epochs")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.show()

