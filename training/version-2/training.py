import tensorflow as tf
assert tf.__version__.startswith('2')

import os
import numpy as np
import shutil
import csv
from pathlib import Path

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

from tensorflow.keras import metrics

epochs = 200
IMAGE_SIZE = 224
BATCH_SIZE = 64
IMG_PATH = Path('img/')
TMP_IMG_PATH = IMG_PATH / 'tmp'

validation = [
  { 'person': '1', 'background': '1', 'outfit': '1' },
  { 'person': '1', 'background': '1', 'outfit': '2' },
  { 'person': '2', 'background': '1', 'outfit': '3' }
]

def main():
  csv_file = open(Path('img/images.csv'))
  csv_reader = csv.reader(csv_file)
  csv_rows = list(csv_reader)
  image_rows = csv_rows[1:]
  labels = set([row[4] for row in image_rows])
  labels.remove('correct')
  # for label in labels:
  #   train_for_label(label, image_rows)
  train_for_label('too_high', image_rows)

def train_for_label(label, image_rows):
  prepare_dirs(label, image_rows)

  (train_generator, val_generator) = get_generators()

  data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("vertical"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.03),
  ])
  
  # image = train_generator[0][0]

  # plt.figure(figsize=(10, 10))
  # for i in range(9):
  #   augmented_image = data_augmentation(image)
  #   ax = plt.subplot(3, 3, i + 1)
  #   plt.imshow(augmented_image[0])
  #   plt.axis("off")
  # plt.show()
  # return 0

  print (train_generator.class_indices)

  labels = '\n'.join(sorted(train_generator.class_indices.keys()))

  with open('labels_%s.txt' % label, 'w') as f:
    f.write(labels)

  IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

  # Create the base model from the pre-trained model MobileNetV2 
  base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                include_top=False, 
                                                weights='imagenet')

  base_model.trainable = False

  model = tf.keras.Sequential([
    data_augmentation,
    base_model,
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(2, activation='softmax')
  ])

  model.compile(optimizer=tf.keras.optimizers.Adam(), 
                loss='categorical_crossentropy', 
                metrics=[metrics.categorical_accuracy])

  history = model.fit(train_generator, 
                      steps_per_epoch=len(train_generator), 
                      epochs=epochs, 
                      validation_data=val_generator, 
                      validation_steps=len(val_generator))

  model.summary()

  plot_accuracy_and_loss(history, label)
  plot_roc(model, train_generator, val_generator, label)


  saved_model_dir = 'save/%s%d' % (label, epochs)
  tf.saved_model.save(model, saved_model_dir)

  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
  tflite_model = converter.convert()

  with open('model_%s.tflite' % label, 'wb') as f:
    f.write(tflite_model)

def get_generators():
  datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rescale=1./255)
      # validation_split=0.1)

  train_generator = datagen.flow_from_directory(
      TMP_IMG_PATH / 'train',
      target_size=(IMAGE_SIZE, IMAGE_SIZE),
      batch_size=BATCH_SIZE)

  val_generator = datagen.flow_from_directory(
      TMP_IMG_PATH / 'validation',
      target_size=(IMAGE_SIZE, IMAGE_SIZE),
      batch_size=BATCH_SIZE)
  
  return (train_generator, val_generator)

def prepare_dirs(label, image_rows):
  try:
    shutil.rmtree(TMP_IMG_PATH)
  except FileNotFoundError:
    print('Didn\'t remove the dir. It didn\'t exist')
  os.mkdir(TMP_IMG_PATH, mode=0o777)
  os.mkdir(TMP_IMG_PATH / 'train', mode=0o777)
  os.mkdir(TMP_IMG_PATH / 'train' / 'correct', mode=0o777)
  os.mkdir(TMP_IMG_PATH / 'train' / label, mode=0o777)
  os.mkdir(TMP_IMG_PATH / 'validation', mode=0o777)
  os.mkdir(TMP_IMG_PATH / 'validation' / 'correct', mode=0o777)
  os.mkdir(TMP_IMG_PATH / 'validation' / label, mode=0o777)

  for image_row in image_rows:
    if (image_row[1], image_row[2], image_row[3]) in [(v['person'], v['background'], v['outfit']) for v in validation]:
      if image_row[4] in ['correct', label]:
        shutil.copyfile(IMG_PATH / image_row[0], TMP_IMG_PATH / 'validation' / image_row[4] / image_row[0])
    else:
      if image_row[4] in ['correct', label]:
        shutil.copyfile(IMG_PATH / image_row[0], TMP_IMG_PATH / 'train' / image_row[4] / image_row[0])

def plot_accuracy_and_loss(history, label):
  acc = history.history['categorical_accuracy']
  val_acc = history.history['val_categorical_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  # plt.figure(figsize=(8, 8))
  # plt.subplot(2, 1, 1)
  plt.plot(acc, label='Training Accuracy', marker='v')
  plt.plot(val_acc, label='Validation Accuracy', marker='o')
  plt.legend(loc='lower right')
  plt.ylabel('Accuracy')
  plt.ylim([min(plt.ylim()),1])
  plt.title('Training and Validation Accuracy')
  plt.xlabel('epoch')
  filename = '%s_%dep_acc.png' % (label, epochs)
  plt.savefig(Path('plots') / filename)
  plt.clf()

  # plt.subplot(2, 1, 2)
  plt.plot(loss, label='Training Loss', marker='v')
  plt.plot(val_loss, label='Validation Loss', marker='o')
  plt.legend(loc='upper right')
  plt.ylabel('Cross Entropy')
  plt.ylim([0,1.0])
  plt.title('Training and Validation Loss')
  plt.xlabel('epoch')
  filename = '%s_%dep_loss.png' % (label, epochs)
  plt.savefig(Path('plots') / filename)
  plt.clf()
  # plt.show()

def plot_roc(model, train_generator, val_generator, label):
  correct_index = train_generator.class_indices['correct']

  predicted = np.array([])
  actual = np.array([])
  print('correct index %d' % correct_index)
  #generators = [train_generator, val_generator]
  generators = [val_generator] # compute ROC only for validation data
  for generator in generators:
    for batch_index in range(len(generator)):
      image_batch, label_batch = generator[batch_index]
      predicted_batch = model.predict(image_batch)
      actual_batch = label_batch
      predicted = np.concatenate((predicted, predicted_batch[:, correct_index]))
      actual = np.concatenate((actual, actual_batch[:, correct_index]))
    
  auc = roc_auc_score(actual, predicted)
  print('AUC: %.2f' % auc)
  fpr, tpr, thresholds = roc_curve(actual, predicted)
  
  plt.plot(fpr, tpr, color='orange', label='ROC', marker='*')
  plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.title('ROC Curve')
  plt.legend()
  filename = '%s_%dep_roc_auc_%f.png' % (label, epochs, auc)
  plt.savefig(Path('plots') / filename)
  plt.clf()

if __name__ == '__main__':
  main()