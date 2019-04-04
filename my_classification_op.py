from __future__ import absolute_import, division, print_function
import random
import matplotlib.pyplot as plt
# Helper libraries
import numpy as np
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

def main(arg_1, arg2,):
  
  epoch = int(input("Number of epoch: " + "\n>> "))
  
  #import the numners 
  mnist = keras.datasets.mnist


  (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
  class_names = ['0', '1', '2', '3', '4', 
                '5', '6', '7', '8', '9']

  train_images.shape
  len(train_labels)
  train_labels
  test_images.shape
  len(test_labels)

  plt.figure()
  plt.imshow(train_images[0])
  plt.colorbar()
  plt.grid(False)
  plt.show()

  train_images = train_images / 255.0

  test_images = test_images / 255.0

  plt.figure(figsize=(10,10))
  for i in range(25):
      plt.subplot(5,5,i+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(train_images[i], cmap=plt.cm.binary)
      plt.xlabel(class_names[train_labels[i]])
  plt.show()

  print("number of nodes? (128 is standard")
  nodes = int(input("Number of nodes? (Standard is 128) \n>>"))

  model = keras.Sequential([
      keras.layers.Flatten(input_shape=(28, 28)),
      keras.layers.Dense(nodes, activation=tf.nn.relu),
      keras.layers.Dense(10, activation=tf.nn.softmax)
  ])
   
  
  choice = int(input("Choose and optimizer: \n(1)adam \n(2)adadelta\n(3)adagrade \n(4)adamax \n(5)Nadam\n>>"))
  opt = "adam"
  if(choice is 1):
    opt = "adam"
  if(choice is 2):
    opt = "adadelta"
  if(choice is 3):
    opt = "adagrad"
  if(choice is 4):
    opt = "adamax"
  if(choice is 5):
    opt = "nadam"
  
  
  model.compile(optimizer= opt, 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
                
  model.fit(train_images, train_labels, epochs=epoch)
  test_loss, test_acc = model.evaluate(test_images, test_labels)

  print('Test accuracy:', test_acc)
  predictions = model.predict(test_images)

  predictions[0]
  np.argmax(predictions[0])
  test_labels[0]
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

  img = test_images[0]

  print(img.shape)

  # Add the image to a batch where it's the only member.
  img = (np.expand_dims(img,0))

  print(img.shape)
  predictions_single = model.predict(img)

  print(predictions_single)

  plot_value_array(0, predictions_single, test_labels)
  plt.xticks(range(10), class_names, rotation=45)
  plt.show()

  prediction_result = np.argmax(predictions_single[0])
  print(prediction_result)

  i = 0
  while i is not -1:
    i = int(input("Pick a test data result (0-9999) or enter -1 to exit. \n >> "))
    if i != -1:
      plt.figure(figsize=(6,3))
      plt.subplot(1,2,1)
      plot_image(i, predictions, test_labels, test_images)
      plt.subplot(1,2,2)
      plot_value_array(i, predictions,  test_labels)
      plt.show()
    else:
      print("Creating Grid Display...")




  # Plot the first X test images, their predicted label, and the true label
  # Color correct predictions in blue, incorrect predictions in red
  num_rows = 10
  num_cols = 6
  num_images = num_rows*num_cols
  plt.figure(figsize=(2*2*num_cols, 2*num_rows))
  for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
  plt.show()


print("Starting numbers.py script")
main(5,6)
print("numbers.py has ended.")