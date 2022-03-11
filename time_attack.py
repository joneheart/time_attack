import cv2
import numpy as np

# net = cv2.dnn.readNetFromTorch('models/eccv16/the_wave.t7')
net = cv2.dnn.readNetFromTorch('models/instance_norm/udnie.t7')

img = cv2.imread('imgs/me.jpg')

h, w, c = img.shape

img = cv2.resize(img, dsize=(500, int(h / w * 500)))

# img = img[162:513, 185:428]

MEAN_VALUE = [103.939, 116.779, 123.680]
blob = cv2.dnn.blobFromImage(img, mean=MEAN_VALUE)

print(img.shape) # (325, 500, 3)

net.setInput(blob)
output = net.forward()

output = output.squeeze().transpose((1, 2, 0))
output += MEAN_VALUE

output = np.clip(output, 0, 255)
output = output.astype('uint8')

cv2.imshow('output', output)
cv2.imshow('img', img)
cv2.waitKey(0)

