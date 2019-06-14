import numpy
import cv2
import os, os.path
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.models import Sequential
from keras import callbacks
from keras.layers.core import Dropout, Flatten, Activation, Dense
from keras.optimizers import RMSprop, SGD,Adam
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.utils import np_utils, multi_gpu_model
from keras import backend as k
import tensorflow as tf

"""
use cpu to run the training
"""
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"


"""
use gpu to run the training
"""
os.environ["CUDA_VISIBLE_DEVICES"]="0"

k.set_image_dim_ordering('th')

seed = 7
numpy.random.seed(seed)
trainblur_directory = '/mnt/train/bad/'
trainnoblur_directory = '/mnt/train/good/'
#testblur_directory = '/mnt/test/bad/'
#testnoblur_directory = '/mnt/test/good/'
filepath = "/blur/"

num_classes = 2
#########################################################
#loading blurry images
img_data_list1=[]
data_dir_list1 = os.listdir(trainblur_directory)

img_list1=os.listdir(trainblur_direc3.
+tory)
for img in img_list1:
	input_img=cv2.imread(trainblur_directory +  img )
	input_img=numpy.swapaxes(input_img,0,2)

	img_data_list1.append(input_img)

img_data1 = numpy.array(img_data_list1)
img_data1 = img_data1.astype('float32')
img_data1 /= 255

print(img_data1.shape)
num_of_samples1 = img_data1.shape[0]
labels1 = numpy.ones((num_of_samples1,),dtype='int64')
print("length of labels1 is "+str(len(labels1)))
print("labels1 are all "+str(labels1[10]))

##########################################################
#loading none blurry images

img_data_list2=[]
data_dir_list2 = os.listdir(trainnoblur_directory)

img_list2=os.listdir(trainnoblur_directory)
for img in img_list2:
	input_img=cv2.imread(trainnoblur_directory +  img )
	input_img = numpy.swapaxes(input_img,0,2)
	img_data_list2.append(input_img)

img_data2 = numpy.array(img_data_list2)
img_data2 = img_data2.astype('float32')
img_data2 /= 255
print(img_data2.shape)

num_of_samples2 = img_data2.shape[0]
labels2 = numpy.ones((num_of_samples2,),dtype='int64')
labels2[:]=0
print("length of labels2 is "+str(len(labels2)))
print("labels2 are all "+str(labels2[10]))
#######################################################
# Combine the two numpy arrays and shuffle
labels=numpy.concatenate((labels1,labels2),axis=0)
img_data = numpy.concatenate((img_data1,img_data2),axis=0)
Y = np_utils.to_categorical(labels, num_classes)
#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
# Data augmenter
dg = image.ImageDataGenerator(horizontal_flip = True, vertical_flip = True)
###########################################################
# Defining the model
input_shape=img_data[0].shape
#height = 512
#width = 512
#input_shape=(height, width, 3)
with tf.device('/gpu:0'):
    model = Sequential()
# layer 1
    model.add(Conv2D(96,(11,11), activation = 'relu', input_shape=input_shape, strides = (4, 4), padding = 'valid', kernel_initializer = 'glorot_normal'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides = (2,2)))

# layer 2
    model.add(Conv2D(256,(5,5), activation = 'relu', strides = (1,1)))
    model.add(MaxPooling2D(pool_size=(3, 3),strides = (2,2)))

# layer 3
    model.add(Conv2D(384, (3,3), activation = 'relu', strides = (1,1)))

# layer 4
    model.add(Conv2D(384, (3,3), activation = 'relu', strides = (1,1)))

# layer 5
    model.add(Conv2D(256, (3,3), activation = 'relu', strides = (1,1)))
    model.add(MaxPooling2D(pool_size = (3,3), strides = (2,2)))

# layer 6
    model.add(Flatten())
    model.add(Dense(4096, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))


###########################
epochs_num = 100
#learning_rate = 0.1
#decay = 0.001
#adam = Adam(lr=learning_rate, decay = decay)
model.compile(loss='binary_crossentropy', optimizer='sgd',metrics=['accuracy'])

model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape
model.layers[0].output_shape
model.layers[0].get_weights()
numpy.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable

# Training
#model.load_weights('motionblur.h5')
es = EarlyStopping(monitor = 'val_acc', patience = 30, verbose = 1)
cp = ModelCheckpoint('noobnet_weights2.h5', monitor = 'val_acc', verbose = 1, save_best_only = True, save_weights_only = True)
model.fit_generator(dg.flow(X_train, y_train), samples_per_epoch = 3000, nb_epoch = epochs_num, validation_data = dg.flow(X_test, y_test), callbacks = [es, cp])
#hist = model.fit(X_train, y_train, batch_size=32, epochs=epochs_num, validation_data=(X_test, y_test), verbose = 1)

filename='model_train_new.csv'
csv_log=callbacks.CSVLogger(filename, separator=',', append=False)


# save every 10 epochs
#checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period = 10)

#callbacks_list = [checkpoint]

# Evaluating the model

score = model.evaluate(X_test, y_test, batch_size = 32, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

test_image = X_test[0:1]
print (test_image.shape)

print(model.predict(test_image))
print(model.predict_classes(test_image))
print(y_test[0:1])

# Save our model here
file = open(filepath+"noobnet_weights2.h5", 'a')
model.save(filepath+"noobnet_weights2.h5")
file.close()
######################################################
