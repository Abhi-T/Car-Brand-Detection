# import libs
from keras.layers import  Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

#re-size
IMAGE_SIZE=[224, 224]

train_path='Datasets/train'
valid_path='Datasets/test'

resnet=ResNet50(input_shape=IMAGE_SIZE+ [3], weights='imagenet', include_top=False)

for layers in resnet.layers:
    layers.trainable= False

#useful for getting number of output classes
folders=glob('Datasets/train/*')

#our layer
X=Flatten()(resnet.output)

prediction=Dense(len(folders), activation='softmax')(X)

#create a model
model=Model(inputs=resnet.input, outputs=prediction)

# print(model.summary())

#tell the model what cost and optimization method to use
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#use the image data generator to import the images from the dataset
from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

#make sure u provide the same target size as initiated for the image size
training_set=train_datagen.flow_from_directory('Datasets/train', target_size=(224,224), batch_size=32, class_mode='categorical')
test_set=test_datagen.flow_from_directory('Datasets/test', target_size=(224,224), batch_size=32, class_mode='categorical')

#fir the model
# r=model.fit_generator(training_set, validation_data=test_set, epochs=50, steps_per_epoch=len(training_set), validation_steps=len(test_set))
r=model.fit_generator(training_set, validation_data=test_set, epochs=5, steps_per_epoch=len(training_set), validation_steps=len(test_set))

print(r.history)

#plot the loss
plt.plot(r.history['loss'], label='train_loss')
plt.plot(r.history['val_loss'] , label='val_loss')

plt.legend()
plt.show()
plt.savefig('LossVal_loss')

#plot the accuracy
plt.plot(r.history['acc'], label='train acc')
plt.plot(r.history['val_acc'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

#save the weights
from keras.models import load_model
model.save('model_resnet50.h5')

y_pred=model.predict_generator(test_set)
print(y_pred)

import numpy as np
y_pred=np.argmax(y_pred, axis=1)
print(y_pred)


from keras.models import load_model
from keras.preprocessing import  image

model=load_model("model_resnet50.h5")

img=image.load_img('Datasets/Test/lamborghini/11.jpg', target_size=(224,224))
x=image.img_to_array(img)
print(x)
x=x/255

x=np.expand_dims(x, axis=0)
img_data=preprocess_input(x)
print(img_data.shape)

model.predict(img_data)

a=np.argmax(model.predict(img_data), axis=1)

print(a)
