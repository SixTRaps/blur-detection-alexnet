import cv2
import os
import numpy as np
import math
from PIL import Image


BASEPATH = '/'
IMAGEPATH = os.path.join(BASEPATH+'data/')
IMAGESAVE = os.path.join(BASEPATH+'slice/')

WIDTH = 512
HEIGHT = 512
IMGCOUNT = 0

"""
Params Description:
    img_data: image data that is read and transform to array.
    n: width of new image will be (n*512)
    m: height of new image will be (m*512)
    i: id1 of sub-picture
    j: id2 of sub-picture
"""
print("Image processing...")
img_queue = os.listdir(IMAGEPATH)
for img in img_queue:
    input_img = cv2.imread(IMAGEPATH + img)
    input_img1 = Image.open(IMAGEPATH + img)
    img_data = np.array(input_img)
    img_data = img_data.astype('float32')
    originalHeight = img_data.shape[0]
    originalWidth = img_data.shape[1]

    print(img+': ')
    print("%f * %f"%(originalWidth,originalHeight))

    if (originalHeight != HEIGHT or originalWidth != WIDTH):
        IMGCOUNT+=1
        n = int(math.ceil(float(originalWidth)/float(WIDTH)))
        m = int(math.ceil(float(originalHeight)/float(HEIGHT)))
        new_image = Image.new("RGBA", (int(WIDTH*n), int(HEIGHT*m)), (0,0,0)) # Generate a new picture with zero paddings.

        # Define margin of paste box.
        """for osX
        left = int((WIDTH*n-originalHeight)/2)
        up = int((HEIGHT*m-originalWidth)/2)
        right = int(left+originalHeight)
        down = int(up+originalWidth)
        """

        left = int((WIDTH*n-originalWidth)/2)
        up = int((HEIGHT*m-originalHeight)/2)
        right = int(left+originalWidth)
        down = int(up+originalHeight)

        new_image.paste(input_img1,(left,up,right,down))  # Paste the original picture on the background

        if not os.path.exists(IMAGESAVE):
            os.makedirs(IMAGESAVE)
        # Crop and save.
        i = 0
        j = 0
        for i in xrange(n):
            for j in xrange(m):
                new_img = new_image.crop((i*WIDTH,j*HEIGHT,(i+1)*WIDTH,(j+1)*HEIGHT))
                new_img_name = img.rstrip('.jpg') +'_%(idx1)s_%(idx2)s'%{'idx1':i, 'idx2':j}+'.jpeg'
                new_img_convert = new_img.convert("RGB")
                new_img_convert.save(IMAGESAVE+new_img_name)
    else:
        input_img1.save(IMAGESAVE+img)
        print(img+": no need to slice")
        print("%f * %f"%(originalWidth,originalHeight))
        IMGCOUNT+=1
    #input_img1.close()

    print("Image process successfully!")
print(IMGCOUNT)
