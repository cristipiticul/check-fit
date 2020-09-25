# pentru a vizualiza filtrele, folositi comanda "tensorboard --logdir logs/summary/blabla"
import tensorflow as tf
assert tf.__version__.startswith('2')

IMAGE_SIZE = 224
BATCH_SIZE = 64
TRAINING_CATEGORY = 'background_fara_unghi_ciudat'
BASE = 'poze/%s/%%s/plank/' % TRAINING_CATEGORY

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255)

train_generator = datagen.flow_from_directory(
    BASE % 'train',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE)

val_generator = datagen.flow_from_directory(
    BASE % 'validation',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=100000)

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                              include_top=False, 
                                              weights='imagenet')

base_model.trainable = False

for image_batch, label_batch in val_generator:
    layer = base_model.get_layer(index=2)
    model2 = tf.keras.Model(base_model.inputs, layer.output)
    res = model2.predict(image_batch)
    print(res.shape)
    logdir = 'logs/summary/blabla'
    file_writer = tf.summary.create_file_writer(logdir)
    with file_writer.as_default():
      tf.summary.image("orig", image_batch, step=0)
      for i in range(0, 32):
        tf.summary.image("filt", res[:,:,:,i:i+1], step=i)
    break