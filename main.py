import os
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
folder_path = 'UTKFace'
age = []
gender = []
img_path = []
for file in os.listdir(folder_path):
    age.append(int(file.split('_')[0]))
    gender.append(int(file.split('_')[1]))
    img_path.append(file)
df = pd.DataFrame({"age":age , "gender":gender,"img":img_path})
train = df.sample(frac=1,random_state=0).iloc[:20000]
test = df.sample(frac=1,random_state=0).iloc[20000:]
#data augmentation

train_datagen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=30,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
# generators

train_generator = train_datagen.flow_from_dataframe(train,
                                                   directory=folder_path,
                                                   x_col='img',
                                                   y_col=['age','gender'],
                                                   target_size=(200,200),
                                                   class_mode='multi_output')

test_generator = test_datagen.flow_from_dataframe(test,
                                                directory=folder_path,
                                                x_col='img',
                                               y_col=['age','gender'],
                                               target_size=(200,200),
                                               class_mode='multi_output')
from keras.applications.vgg16 import VGG16
from keras.layers import *
from keras.models import Model
vggnet = VGG16(include_top = False,input_shape=(200,200,3))
vggnet.trainable=False

output = vggnet.layers[-1].output

flatten = Flatten()(output)

dense1 = Dense(512,activation='relu')(flatten)
dense2 = Dense(512,activation='relu')(flatten)

dense3 = Dense(512,activation='relu')(dense1)
dense4 = Dense(512,activation='relu')(dense2)

output1 = Dense(1,activation='linear',name='age')(dense3)
output2 = Dense(1,activation='sigmoid',name='gender')(dense4)
model = Model(inputs=vggnet.input,outputs=[output1,output2])
model.summary()
from keras.utils import plot_model
plot_model(model)
model.compile(optimizer='adam',loss={'age':'mae','gender':'binary_crossentropy'},metrics={'age':'mae','gender':'accuracy'},loss_weights={'age':1,'gender':99})
model.fit(train_generator,batch_size=32,epochs=10,validation_data=test_generator)
from keras.preprocessing import image
import numpy as np

img_path = 'path_to_your_image.jpg'  
img = image.load_img(img_path, target_size=(200, 200))  
img_array = image.img_to_array(img)  
img_array = np.expand_dims(img_array, axis=0) 

img_array = img_array / 255.0  

age_pred, gender_pred = model.predict(img_array)

predicted_age = int(age_pred[0]) 
predicted_gender = "Male" if gender_pred[0] > 0.5 else "Female"  

print("Predicted Age:", predicted_age)
print("Predicted Gender:", predicted_gender)