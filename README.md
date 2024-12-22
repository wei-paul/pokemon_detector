# Knowledge of everything you need to know about this project
## Train.py

### Overview of the libraries imported
Tensorflow: a powerful open-source machine learning library developed by Google
- supports a wide range of machine learning and deep learning models. 
- can run on multiple CPUs and GPUs, making it suitable for large-scale machine learning tasks.
- can automatically calculate gradients, which is crucial for training neural networks.
- TensorBoard for visualizing the learning process and model architecture.
- TensorFlow is optimized for speed and efficiency in numerical computations.
In this project TensorFlow serves as the backend for Keras, providing the computational engine for building and training your neural network model. Tensorflow handles the low-level computations and optimizations

Keras: High-level neural network API
- Runs on top of TensorFlow
- Used to build and train your autoencoder model
- Provides layers and model structures for easy network construction
**Keras layers**
- Building blocks for your neural network
- Used to construct the encoder and decoder parts of your autoencoder
**Keras backend**
- Interface to the underlying TensorFlow operations
- Allows for more advanced manipulations if needed
**Keras models**
- Used to define the overall structure of your autoencoder

Numpy: Handles numerical operations on arrays
- Used for data manipulation and preprocessing

OpenCV (cv2): Used for image loading, resizing, and color space conversion

Matplotlib: For visualizing images and potentially plotting training progress

os: Python module that provides a way to interact with the operating system
- os module helps manage system-level interactions within your Python script, enhancing its ability to work with files and system settings.

### Code and steps explained
```
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
```
- we first need to configure GPU usage and resolve OpenMP conflicts. This line (cuda_visible_devices) specifies which GPU device to use. Setting it to "0" tells the system to use the first available GPU. This is useful for controlling GPU allocation, especially in multi-GPU systems.
-  The second line allows multiple OpenMP runtime libraries to coexist. It's used to resolve conflicts that can occur when different libraries (like TensorFlow and its dependencies) try to initialize their own OpenMP runtimes. This helps prevent crashes and ensures smoother execution of the code.
```
x_train = []
for t in range(925):
    image = cv2.imread('./Train/pokemon ('+str(t+1)+').png')
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA) **This code important**
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x_train.append(img_rgb)
x_train = np.array(x_train)
```
- This code block is performing image preprocessing for the training dataset
- Initializing an empty list x_train to store preprocessed images.
- Loop through 925 images 
- For each image, first read the file, then resize the image to 64 x 64 with INTER_AREA interpolation
- In this case, INTER_AREA interpolation ensures a high-quality resizing of the pokemon images to the target size of 64x64 pixels, maintaining as much visual information as possible in the smaller format.
- Convert the image from BGR to RGB color space using cv2.cvtColor()
- Append each preprocessed image to the x_train list.
- After processing all images, convert the list to a NumPy array for efficient computation.
- Summary: This preprocessing ensures all images are of uniform size and color format, preparing them for input into the autoencoder model. The resulting x_train array will have shape (925, 64, 64, 3) (4D array), representing 925 images of size 64x64 with 3 color channels.
```
# plt.imshow(x_train[0].reshape(64, 64, 3))
# plt.show()
```
- The commented out code is for visualizing purposes 
- It would display the first image in the x_train array.
- plt refers to matplotlib.pyplot, a plotting library in Python. The imshow function is used to display an image, and show() renders the plot.
- This code would display the first preprocessed Pokemon image from the dataset, reshaped to 64x64 pixels with 3 color channels (RGB). It's a useful way to visually verify that your image preprocessing steps are working correctly.
```
x_train = x_train.astype('float32')/255
```
- This code here normalizes the pixel values in the x_train array. 
- Converts the data type to float32, which is more suitable for neural network computations.
- Divides all pixel values by 255, scaling them from the original range of 0-255 to 0-1.
- Crucial for neural networks as it helps in faster convergence during training and ensures that all input features are on a similar scale. It's a standard preprocessing technique in image-based machine learning tasks.
```
# print('x_train shape:', x_train.shape)
img_shape = x_train.shape[1:] 
```
- The commented-out print statement would display the shape of the x_train array, which is useful for verifying the dimensions of your preprocessed dataset. (expecting 925, 64, 64, 3)
- The line img_shape = x_train.shape[1:] extracts the shape of a single image from the x_train array. It takes all dimensions except the first one, which would be (64, 64, 3).
```
input_img = keras.Input(shape=img_shape)       
print(input_img.shape) 
```
- This code creates the input layer for the autoencoder model using Keras.
- The keras.Input() function defines the shape of the input data the model will accept. Here, img_shape specifies the dimensions of a single image (likely 64x64x3 based on earlier preprocessing).
- In Keras, Input() creates a symbolic tensor that serves as the entry point for your neural network model.
- Creating an input tensor with keras.Input() defines the shape and type of data your model expects, setting the stage for the subsequent layers in your autoencoder architecture.
```
Architecture
#####
x = Conv2D(64, (3, 3), padding='same')(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
encoded = MaxPooling2D((2, 2), padding='same', name='encoder')(x)

x = Conv2D(16, (3, 3), padding='same')(encoded)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(3, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
decoded = Activation('sigmoid')(x)
```
- We now set up our autoencoder architecture, this code sets up the layers and stages through which the data (input tensor) passes
- Conv2D(64, (3, 3), padding='same')(input_img) Applies 64 convolutional filters of size 3x3 to the input image
'padding='same'' maintains the spatial dimensions
- BatchNormalization()(x) Normalizes the outputs of the previous layer
- Activation('relu')(x) Applies the ReLU activation function, introducing non-linearity
- MaxPooling2D((2, 2), padding='same')(x) Reduces spatial dimensions by half, keeping the maximum value in each 2x2 window
- Conv2D(32, (3, 3), padding='same')(x) Applies 32 convolutional filters, reducing the number of feature maps
- The next few lines repeat the pattern of Conv2D, BatchNormalization, Activation, and MaxPooling, progressively reducing spatial dimensions and number of filters
- encoded = MaxPooling2D((2, 2), padding='same', name='encoder')(x) Final encoding step, further reducing spatial dimensions

- The next few lines form the decoder, reversing the encoding process:
- Conv2D layers increase the number of filters
  UpSampling2D layers increase spatial dimensions
  BatchNormalization and ReLU activation are applied after each Conv2D
- x = Conv2D(3, (3, 3), padding='same')(x) Final convolutional layer to produce a 3-channel output (RGB)
- decoded = Activation('sigmoid')(x) Sigmoid activation ensures output values are between 0 and 1, suitable for image pixel values

```
autoencoder = Model(input_img, decoded)
autoencoder.summary()
```
- Creates the autoencoder model using Keras' Model class, specifying the input and output layers.
- Prints a summary of the model architecture.
```
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
```
- Compiles the model with Adam optimizer and binary cross-entropy loss function.
```
autoencoder.fit(x_train, x_train,
                epochs=200,
                batch_size=32,
                shuffle=True,
                verbose=2)

autoencoder.save('model.h5')
```
- Trains the autoencoder for 200 epochs with a batch size of 32. The input and output are both x_train, as the autoencoder aims to reconstruct the original images.
- Saves the trained model as a .h5 file for later use.





### New concepts/code/rabbitholes
**What is tensor/tensorflow** - It's essentially a multi-dimensional array of data
- Tensors can represent various types of data: scalars (0D), vectors (1D), matrices (2D), or higher-dimensional arrays.
- For image data, a tensor typically has 4 dimensions: (batch_size, height, width, channels).
- TensorFlow, as the name suggests, is built around the concept of tensor operations.
- In Keras, Input() creates a symbolic tensor that serves as the entry point for your neural network model.
- These tensors flow through the layers of the neural network, undergoing transformations at each step.

**Why we use CNN**
- We want to reduce spatial dimensions while retaining important features, WHICH CNN is good at
- The convolutional layers in CNNs can effectively capture spatial hierarchies and local patterns in images, which is crucial for tasks like image compression and reconstruction.
- The advantage of parameter sharing, which reduces the number of learnable parameters compared to fully connected networks, making them more efficient for image processing tasks.
- The use of pooling layers in the encoder and upsampling in the decoder allows the network to effectively compress and decompress the image information, which is the core functionality of an autoencoder.

**OpenMP (Open Multi-Processing)** - is a runtime libraries are components that support parallel programming in shared memory multiprocessing environments.
- Thread management: Creating, synchronizing, and terminating threads.
- Work scheduling: Distributing tasks among available threads.
- Data sharing: Managing shared and private variables in parallel regions.
- Synchronization: Implementing barriers and locks for coordinated execution.
- In the context of deep learning frameworks like TensorFlow, OpenMP is often used to optimize performance on multi-core CPUs.
- I had to add this line because running this in jupyter notebook gave me the following error:
```
2024-07-31 16:23:48.324749: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
OMP: Error #15: Initializing libiomp5, but found libiomp5md.dll already initialized.
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://openmp.llvm.org/
```
**interpolation=cv2.INTER_AREA**: INTER_AREA is an interpolation method used in OpenCV's resize function
- particularly effective when downsampling (shrinking) images
- method uses pixel area relation for resampling, which often produces better results than simple methods like nearest neighbor interpolation, especially when reducing image size
- helps to minimize aliasing artifacts that can occur during image resizing (which in turn, means better quality training data)
- Different types of interpolation methods for OpenCV image resizing:  
INTER_NEAREST: Nearest-neighbor interpolation. Fast but can result in blocky images. Suitable for enlarging pixel art or when speed is critical.

INTER_LINEAR: Bilinear interpolation. Good balance between speed and quality for both enlarging and shrinking images.

INTER_CUBIC: Bicubic interpolation. Produces smoother edges than bilinear, better for enlarging images.

INTER_LANCZOS4: Lanczos interpolation over 8x8 pixel neighborhood. High-quality downsampling, but computationally expensive.

INTER_LINEAR_EXACT: Bit-exact bilinear interpolation. Useful when exact reproducibility is required.

**Understanding 2D Convolutions**
- https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1


### Useful code for future
- image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA) **important when resizing + maintaining the quality of the original image**

