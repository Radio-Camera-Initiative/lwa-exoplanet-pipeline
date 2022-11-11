import numpy as np
import sys

if len(sys.argv) > 1:
    name = sys.argv[1]
else:
    name = '1_image.npy'

a = np.load(name)
print(a[0].max())
#assert(a[0].max() == 0.0166628360748291)

peak = np.unravel_index(np.argmax(a[0]), a[0].shape)

import matplotlib.pyplot as plt 
# plt.imshow(a[0][peak[0]-100:peak[0]+100, peak[1]-100:peak[1]+100])
plt.imshow(a[0], vmin=-466.29, vmax=642.573)
plt.colorbar()
plt.show()
