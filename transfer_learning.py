import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
tf.__version__

# As we will use ``Tensorflow``, we will have to format the images

train_imagenerator = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input, rotation_range =, width_shift_range =,
                                     height_shift_range =,
                                     zoom_range =,
                                     horizontal_flip =,
                                     vertical_flip =)

# # Transfer Learning:


#loading the model:
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)


x = base_model.output


x = tf.keras.layers.GlobalAveragePooling2D()(x) # to do the Pooling is necessary the last layer

#the dense layers (how many layer you want you will pu the "x", here I put 3 but can be more, you decide):
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
preds = tf.keras.layers.Dense(4, activation='softmax')(x) #here you put the numbers of predictions you need, like you have to predict 4 diferent kind of clasification

model = tf.keras.Model(inputs = base_model.input, outputs = preds)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(generator=train_generator,
                              epochs=25,
                              steps_per_epoch=step_size_train,
                              validation_data = test_generator,
                              validation_steps=step_size_test)


# # The predictions
filenames = test_generator.filenames

predictions = model.predict_generator(test_generator, steps = len(filenames))

predictions2 = []

for i in range(len(predictions)):
    predictions2.append(np.argmax(predictions[i]))

test_generator.class_indices

# # Forecasting using image
image = tf.keras.preprocessing.image.load_img(r'the image you want', target_size=())
plt.imshow(image);

image = tf.keras.preprocessing.image.img_to_array(image)
np.shape(image)

image = np.expand_dims(image, axis = 0)
np.shape(image)

image = tf.keras.applications.resnet50.preprocess_input(image)
predictions = model.predict(image)
print(predictions)