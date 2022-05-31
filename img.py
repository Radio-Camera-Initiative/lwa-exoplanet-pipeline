import numpy as np

a = np.load('image.npy')
print(a[0].max())
#assert(a[0].max() == 0.0166628360748291)

peak = np.unravel_index(np.argmax(a[0]), a[0].shape)

import matplotlib.pyplot as plt 
plt.imshow(a[0][peak[0]-100:peak[0]+100, peak[1]-100:peak[1]+100])
plt.show()
