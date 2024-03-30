# GAN - MNIST Digits
This is an implementation of a general adversarial network in tensorflow, that uses two neural networks, a generator and a discriminator, to generate images that ressemble, in this case the MNSIT digit dataset. Working in tandem, the generator attempts to generate images to fool the discriminator, while the discriminator attempts to identify whether an image was generated by the generator or if it is a real image from the dataset.

## Generator
This neural network uses several layers, including convolutions, to generate images ressembling the given dataset.

## Discriminator
The discriminator neural network uses several layers to attempt to discern images created by the generator and images from the dataset

## Plotting metrics
There is a function called plot_metric that is used to plot generator and discriminator loss, and accuracy.
These are kept track of in the loss log dictionary, where loss is updated for every batch in every epoch
while the accuracy is updated every epoch. 

## Variables and hyperparameters
These can be found near the beginning of the colab or IDE file to be easily changed
* batch size is the number of images in a batch
* epochs is the number of training iterations the model goes through
* noise dimension is the dimensions of the random noise vector that is input for the generator. This also refers to the dimension of the generator input.
* learning rate of discriminator/generator refers to the step size at which model parameters are updated during optimization. In other words, it controls the rate at which the model learns.

## Running the Project in Google Colab
1. Navigate to the `GAN.ipynb` file.
2. Click on the "Open in Colab" button at the top of the file.
3. Ensure that the runtime type is set to Python 3 by clicking on the "Runtime" menu, then "Change runtime type", and selecting "Python 3".
4. if available, change Hardware accelerator to T4 GPU, or a different option as CPU is not fast
5. Run the entire notebook by clicking on the "Runtime" menu and selecting "Run all". Alternatively, run each cell individually by clicking on the play button to the left of each cell.

## Running the project (IDE) - NOT WORKING
To run this project on your machine, it would be worth having an already set up IDE, otherwise using Colab would be easiest.
1. Navigate to the directory where you want the project to be located using ```cd directory_name```
2. Within that directory, create a new folder for the project using ```mkdir project_folder_name```
3. Navigate to the new folder using ```cd project_folder_name```
4. Initialize a git repo using ```git init```
5. Run ```git clone https://github.com/lucash-h/MNIST_GAN```
6. Navigate to cloned repo directory using ```cd MNIST_GAN```
7. Check the dependencies needed using ```cat requirements.txt```
8. Download all dependencies using ```pip install -r requirement.txt``` or download specific dependencuy using ```pip install [specific dependency]```
9. In terminal run commands ```set PYTHONENCODING=utf-8``` followed by ```python GAN.py```

