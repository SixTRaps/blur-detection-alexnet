import cv2
import numpy as np
#from datetime import time
from keras.models import load_model
import os
from keras import backend
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

backend.set_image_dim_ordering('tf')
model = load_model('noobnet_weights.h5')
model.summary()
#start = time.time()
img_list = []
image = cv2.imread("img3.jpeg")
#image = cv2.resize(image, (512,512), interpolation = cv2.INTER_LINEAR)
image = np.swapaxes(image, 0, 2)
img_list.append(image)
img_data = np.array(img_list)
img_data = img_data.astype('float32')
img_data /= 255.0


print(img_data.shape)
predict = model.predict_classes(img_data)

if predict == 1:
    print('Result: The picture is blur.')
else:
    print('Result: The picture is clear.')
"""
end = time.time()
duration = end - start
print('Recognition time: ')
print(duration)
"""
