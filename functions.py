'''
This function builds a generator model
Input: latent_dimension, which is the size of the input noise vector being put into the generator
Output: a model that takes in a noise vector and outputs a generated image
'''

def build_generator(latent_dimension):
  xin = Input((latent_dimension))
  xout = Dense(7*7*128, input_dim = latent_dimension)(xin) #map input noise to higher dimension
  xout = Reshape((7,7,128))(xout) #reshape so suitable for convolution
  xout = BatchNormalization()(xout) #normalizes
  xout = LeakyReLU(alpha=0.2)(xout) #non linearity
  xout = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(xout) #generate higher res
  xout = BatchNormalization()(xout)
  xout = LeakyReLU(alpha=0.2)(xout)
  xout = Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', activation='sigmoid')(xout) #tanh provides higher res -1 -> 1 rather than 0 -> 1 and vals are centered @ 0 for other layers
  return Model(xin,xout)

'''
This function builds a discriminator model
Input: None
Output: a model that takes in an image and outputs a value between 0 and 1, where 0 is fake and 1 is real
'''

def build_discriminator():
    xin = Input((28,28,1))
    xout = Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=(28, 28, 1))(xin)
    xout = LeakyReLU(alpha=0.2)(xout)
    xout = Dropout(0.4)(xout)
    xout = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(xout)
    xout = LeakyReLU(alpha=0.2)(xout)
    xout = Dropout(0.4)(xout)
    xout = Flatten()(xout)
    xout = Dense(1, activation='sigmoid')(xout) #sets vals 0-1 --> fake - real

return Model(xin,xout)

'''
Plot images function is used to plot images, such as the generated images from the generator
Input:
Sample number, the amount of images to plot
images, the images to plot

Output: None
'''
def plot_images(sample_number, images):

    _edge_length = int(np.sqrt(sample_number))
    fig,ax = plt.subplots(nrows = _edge_length, ncols = _edge_length,figsize=(10,10))
    for i in range(_edge_length**2):
        ax[i//_edge_length][i%_edge_length].imshow(images[i, :, :, 0])
        ax[i//8][i%8].axis('off')



'''
plot metrics function is used to plot metrics of the model, such as accuracy and loss
It takes in the following parameters:
two lists and their titles

downsample: boolean value that determines whether to downsample the data
ds_value: the value to downsample by

x_title: the title of the x-axis
y_title: the title of the y-axis
plot_title: the title of the plot
'''

def plot_metrics(list1,list1_title, list2, list2_title, downsample=False, ds_value=1, x_title='X', y_title='Y', plot_title='Plot'):

    if downsample == True:
        list1 = list1[::ds_value]
        list2 = list2[::ds_value]

    plt.figure().set_figwidth(15)
    plt.grid(True)

    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(plot_title)

    plt.plot(list1, label=list1_key)
    plt.plot(list2, label=list2_key)

    plt.legend()
    plt.show()