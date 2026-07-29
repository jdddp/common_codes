import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

# V: [Range, Beam]
df = pd.read_csv("./mblyy/100.csv", header=None)

# 转成 numpy
V = df.to_numpy().astype(np.int8)

profile = np.mean(V, axis=1)

plt.figure(figsize=(10,4))
plt.plot(profile, linewidth=2)

plt.xlabel("Range Bin")
plt.ylabel("Mean Echo Intensity")
plt.title("Mean Echo Intensity vs Range")

plt.grid(True)
plt.tight_layout()
plt.show()

# cv2.imshow('img', V)
# cv2.waitKey(0)
# cv2.destroyAllWindows()