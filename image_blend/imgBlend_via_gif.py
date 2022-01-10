import cv2
import imageio
import numpy as np

fg = cv2.resize(cv2.imread('TS.png'), (480, 270))
mask = cv2.resize(cv2.imread('TaylorSwift.png', cv2.IMREAD_UNCHANGED), (480, 270))

url = 'https://media0.giphy.com/media/chWuD9atZtWwegRY18/giphy.gif'
frames = imageio.mimread(imageio.core.urlopen(url).read(), '.gif')

# fg_h, fg_w, fg_c = fg.shape
# bg_h, bg_w, bg_c = frames[0].shape
# top = int((bg_h-fg_h)/2)
# left = int((bg_w-fg_w)/2)
# bgs = [frame[top: top + fg_h, left:left + fg_w, 0:3] for frame in frames]

results = []
alpha = 0.6
for i in range(len(frames)):
    result = fg.copy()
    frames[i] = cv2.cvtColor(frames[i], cv2.COLOR_BGRA2BGR)
    result[mask[:,:,3] != 0] = alpha * result[mask[:,:,3] != 0]
    frames[i][mask[:,:,3] == 0] = 0
    frames[i][mask[:,:,3] != 0] = (1-alpha)*frames[i][mask[:,:,3] != 0]
    result = result + frames[i]
    results.append(result)

# imageio.mimsave('result1.gif', results)
print(np.array(results).shape)