import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import ipdb
st = ipdb.set_trace
img = np.zeros((400,700,3)).astype(np.float32)
st()
img_cir = cv2.circle(img, (200,200), radius=30, color=(255,0,0), thickness=-1)
plt.imshow(img)
plt.show(block=True)
