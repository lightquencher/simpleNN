import numpy as np
rands1 = np.add(1, np.multiply(2, np.random.rand(1000)))
rands2 = np.add(1, np.multiply(2, np.random.rand(1000)))
with open("data_flowers.txt", 'w') as file:
    for i in range(1000):
      file.write("{0:.2f} {1:.2f} {2}\n".format(rands1[i], rands2[i], ("1 0" if rands1[i]<rands2[i] else "0 1")))