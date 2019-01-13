

#TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

#Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


#loading data into the right names
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# as the dataset does not contain 'names' for the labels, we define and store them separately
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Explore the format of the dataset
print(train_images.shape)# shows 60000 images, each in 28x28
#print(train_images)
print(len(train_labels))# and accordingly, shows there are 60000 labels in the training set

print(train_labels)# each training label is an integer between 0 and 9
#array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)

print(test_images.shape)# 10000 test images and each 28x28
#(10000,28,28)
print(len(test_labels))# and accordingly 10000 test labels
#10000



##### Preprocessing data
plt.figure()
plt.imshow(train_images[0])
#plt.show()
plt.colorbar()
plt.grid(False)


train_images = train_images/255.0
test_images = test_images/255.0

plt.figure(figsize=(10,10))

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    
    plt.xlabel(class_names[train_labels[i]])


### Building the model
#setting up the layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

#compiling the model
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#training the model
model.fit(train_images, train_labels, epochs=5)

#Evaluating accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels) # overfitting !
print('Test accuracy:', test_acc);

#predicting
predictions = model.predict(test_images)
print(predictions[0]) # gives out predictions of the first element as an array of 10 numbers, out put by the last layer of the NN
print(np.argmax(predictions[0]))# retrieving the max number(prediction value(confidence!))
print(test_labels[0])# test labels reflects the truth!

#Graphing it up

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


# testing individual predictions with graphs
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)


i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)



# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)

# Grab an image from the test dataset
img = test_images[0]

print(img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)

predictions_single = model.predict(img)

print(predictions_single)


plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

print(np.argmax(predictions_single[0]))

plt.show()
    






