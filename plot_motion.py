import numpy as np
import matplotlib.pyplot as plt

data = np.load("landmarks/Accommodation.npy")   # change to your file name
print("Shape:", data.shape)

plt.plot(data[:, 99], label="left hand wrist (x)")
plt.plot(data[:, 162], label="right hand wrist (x)")
plt.xlabel("frame")
plt.ylabel("horizontal position")
plt.legend()
plt.show()