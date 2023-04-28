import matplotlib.pylab as plt
import numpy as np

bc=np.load("bc.npy")



radius=50


fig ,(ax1,ax2)= plt.subplots(1,2,sharey='row', figsize=(10,5))

plt.subplots_adjust(wspace=0, hspace=0)

circle = plt.Circle((0, 0), radius, edgecolor='black', fill=False)
ax1.add_patch(circle)

for i in range(len(bc)):

    ax1.plot(bc[i,:, 0, 0], bc[i,:, 0, 1])

ranges=radius*1.9

ax1.set_xlim([-ranges, ranges])
ax1.set_ylim([-ranges, ranges])

ax1.set_title("BC", y=1.0, pad=-20,fontsize=20)

circle = plt.Circle((0, 0), radius, edgecolor='black', fill=False)
ax2.add_patch(circle)

ccil=np.load("ccil.npy")

for i in range(len(ccil)):
    ax2.plot(ccil[i, :, 0, 0], ccil[i, :, 0, 1])

ax2.set_xlim([-ranges, ranges])
ax2.set_ylim([-ranges, ranges])
ax2.set_title("CCIL", y=1.0, pad=-20,fontsize=20)


plt.show()
