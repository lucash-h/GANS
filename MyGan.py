import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob
import time
from tensorflow.keras import layers
import time
from tensorflow.keras.layers import Dense, Conv2D, Dropout, LeakyReLU, Flatten, Reshape, BatchNormalization, Conv2DTranspose, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tqdm import tqdm

from functions import *

def main():

  '''All the declared parameters of this program '''
  batch_size = 32
  latent_dimension = 100
  noise_dimension = 100
  epochs = 5

  generator_learning_rate = 0.001
  discriminator_learning_rate = 0.001


  (real_images, real_labels), (test_set,test_set_labels) = tf.keras.datasets.mnist.load_data()

  assert real_images.shape == (60000, 28, 28)
  assert real_labels.shape == (60000,)

  assert test_set.shape == (10000, 28, 28)
  assert test_set_labels.shape == (10000,)

  real_images = real_images / 255.0
  test_set = test_set / 255.0


  generator = build_generator(latent_dimension,)
  #generator.compile(optimizer='Adam', loss = 'bce')

  discriminator = build_discriminator()
  #discriminator.compile(optimizer='Adam', loss='bce', metrics=['accuracy'])

  generator.summary()
  discriminator.summary()

  discriminator.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss=tf.keras.losses.BinaryCrossentropy(),
      metrics=['accuracy']
  )

  generator.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss=tf.keras.losses.BinaryCrossentropy(),
      metrics=['accuracy']
  )

  bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

  gen_optimizer = tf.keras.optimizers.Adam(learning_rate=generator_learning_rate)
  disc_optimizer = tf.keras.optimizers.Adam(learning_rate=discriminator_learning_rate)

  @tf.function
  def train_step(images):
  # Training loop
    noise = tf.random.normal([batch_size, noise_dimension])
    loss_log = {'generator_loss':[],'discriminator_loss':[], 'real_discriminator_loss':[], 'fake_discriminator_loss':[]} #, 'real_disc_acc':[], 'fake_disc_acc':[]}

    with tf.GradientTape() as gen_tape:
      generated_images = generator(noise)

      fake_output = discriminator(generated_images)
      generator_loss = bce(tf.ones(batch_size),fake_output)
      loss_log['generator_loss'].append(generator_loss)


    gradients_of_generator = gen_tape.gradient(generator_loss, generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    with tf.GradientTape() as disc_tape:
      real_output = discriminator(images)
      fake_output = discriminator(generated_images)

      fake_discriminator_loss = bce(tf.zeros(batch_size),fake_output)
      loss_log['fake_discriminator_loss'].append(fake_discriminator_loss)

      real_discriminator_loss = bce(tf.ones(batch_size),real_output)
      loss_log['real_discriminator_loss'].append(real_discriminator_loss)

      total_discriminator_loss = fake_discriminator_loss + real_discriminator_loss
      loss_log['discriminator_loss'].append(total_discriminator_loss)

    gradients_of_discriminator = disc_tape.gradient(total_discriminator_loss, discriminator.trainable_variables)
    disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))



    return loss_log

    loss_dict = {'generator_loss':[],
              'discriminator_loss':[], 
              'real_discriminator_loss':[], 
              'fake_discriminator_loss':[],
              'real_discriminator_acc':[],
              'fake_discriminator_acc':[]}

  #split test set into batches of 32
  num_batches = len(test_set) // batch_size
  split_indices = [i * batch_size for i in range(1, num_batches)]
  test_set_batched = np.split(test_set, split_indices)

  print(len(test_set_batched))


  for epoch in range(epochs):
    for batch in tqdm(real_images):
      batch_loss = train_step(batch)

      for key in batch_loss:
        loss_dict[key].append(batch_loss[key])
    #add accuracy
    batch_index = np.random.randint(0, 311) #chooses random int value from 0 to 311 to pull random batch from test set
    result_real_eval = discriminator.evaluate(x=test_set_batched[batch_index],
                                    y=tf.ones(32), 
                                    verbose = 0)
    
    result_gen_eval = discriminator.evaluate(x=generator(tf.random.normal([batch_size, noise_dimension])),
                                    y=tf.zeros(32), 
                                    verbose = 0)
    
    loss_dict['real_discriminator_acc'].append(result_real_eval[1])
    loss_dict['fake_discriminator_acc'].append(result_gen_eval[1])


  generated = generator(tf.random.normal((n_generated_samples,100)))

  plot_images(64, generated)


  plot_metrics(loss_dict['generator_loss'], 
              'Generator Loss', 
              loss_dict['discriminator_loss'], 
              'Discriminator Loss', 
              True, 
              10, 
              'X', 
              'Overall Loss', 
              'Generator vs Discriminator Loss')

  plot_metrics(loss_dict['real_discriminator_loss'], 
              'Real Discriminator Loss', 
              loss_dict['fake_discriminator_loss'], 
              'Fake Discriminator Loss', 
              True, 
              10, 
              'X', 
              'Discriminator Loss Quality', 
              'Real vs Fake Discriminator Loss')

  plot_metrics(loss_dict['real_discriminator_acc'],
              'Real Discriminator Accuracy',
              loss_dict['fake_discriminator_acc'],
              'Fake Discriminator Accuracy',
              False,
              10,
              'X',
              'Discriminator Accuracy Quality',
              'Real vs Fake Discriminator Accuracy')

if __name__ == "__main__":
    main()