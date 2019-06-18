import cv2
import numpy as np
#from datetime import time
from keras.models import load_model
import os
from keras import backend
from PIL import Image
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#BASEPATH = ""
IMAGEPATH = "./slice/"

backend.set_image_dim_ordering('tf')
model = load_model('noobnet_weights.h5')
model.summary()
#start = time.time()
with open('predict_prob.txt','w') as file:
    for img in os.listdir(IMAGEPATH):
        img_list = []
        image = cv2.imread(IMAGEPATH + img)
        print(IMAGEPATH+img)
        image = np.swapaxes(image, 0, 2)
        img_list.append(image)
        img_data = np.array(img_list)
        img_data = img_data.astype('float32')
        img_data /= 255.0

        predict = model.predict_classes(img_data)

        predict_data = model.predict(img_data)
        prob = predict_data[0][0]
        p_c = r"clear"
        p_b = r"blur"
        if re.search(re.compile(p_c), img)!=None:
            label = re.search(re.compile(p_c), img).group(0)
        elif re.search(re.compile(p_b), img)!=None:
            label = re.search(re.compile(p_b), img).group(0)

        file.write('%s %f %s'%(img,prob,label) + '\n')

        if predict == 1:
            print('Result: Picture %s is blur. Prob: %f'%(img, prob))
        else:
            print('Result: Picture %s is clear. Prob: %f'%(img, prob))
        """
        end = time.time()
        duration = end - start
        print('Recognition time: ')
        print(duration)
        """
file.close()
