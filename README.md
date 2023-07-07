# Image Type Classification Documentation
## About the Use Case

The database typically contains a vast set of images that are unlabelled and/or disorganised, and they contain vital information including Medical Images, Medical Documents and Clinical Photographs. Some examples of such images include the following (these were also the selection of images used to train and create the program):

1.	**Medical Images** 
a.	X Rays
b.	CT Scan Brain (Only Brain is good enough to start with as it will form the major part of CT Image universe)
c.	MRI Spine (Only Spine is good enough to start with as it will form major part of MRI Image Universe)
d.	Angiogram showing Blockages/Stents.
e.	MRI Scan Reports
f.	Angiography/Angioplasty Reports
g.	Echocardiography Reports
h.	Implant / Device Invoice
i.	Implant / Device Barcode Stickers

2.	**Medical Documents (Printed, Handwritten or a combination)**
a.	Clinical Notes
b.	Lab Reports
c.	X Ray Reports
d.	Ultrasound Examination Reports
e.	CT Scan Reports
f.	Discharge Notes

3.	**Clinical Photographs**
a.	Injury Photos
b.	Burn Photos
c.	Intraoperative Photographs

These images/documents are important for a variety of purposes, few examples being documentation and evidence of present and historical cases and claims, and information that could be used to diagnose/re-diagnose conditions.

Tagging and parsing such images into the aforementioned categories is important for indexing and cataloguing, easy referencing and for carrying out further Computer Vision/Image Analytics specialised programs particularly at the large scale of the NHA database. Some of these specialised computer vision use cases could include examples such as diagnosing diseases or issues in X-Rays, Mining and verifying data from prescriptions or other claim documents and running Facial Recognition to verify identity, amongst a vast sea of other potential use cases.

**Note:** The production version of this program may have been programmed to convert PDFs to images for use with the existing model, however this is out of the scope of this documentation.

## About the Program

The model being used in this program is a simple CNN model built using Keras with a Tensorflow backend (v2.7.0) in Python 3. It contains 3 convolutional layers and 3 dense/fully connected layers with an input shape of 128x128x3 and an output shape of 3. Code used to create the model:

model = tf.keras.models.Sequential()

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(128,128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))

model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))

model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))

model.add(Flatten())

model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

This model takes an input of an RGB colour image as a NumPy array of shape (128, 128, 3) (after being normalised) and outputs a probability array [x, y, z] where x, y and z represent the probability that the image is a Medical Image, Medical Document or Clinical Photograph respectively. The highest probability will represent that class/category that the model will identify an image into.

## What the Model can do

The model being applied can categorise images into the aforementioned categories with about a 98% accuracy under test conditions. It is good at handling misaligned/rotated or inverted images, slightly blurry images that are typically taken through scanners or smartphones.

It is designed to take in any unlabelled image encountered in the NHA image data universe provided at the time of creation of this model.

## What the Model cannot do

As with most CNN models, this model has a singular purpose. It is designed to take an input image and output a probability array corresponding to the aforementioned categories in the order specified. It has no intelligence to notify the user if an image is unclear, has a mixture of categories or if it is unsure of which category an image would belong to (although this could be gauged to some degree by looking at the probabilities in the output array). It will output the prediction array even if the images input have nothing to do with the aforementioned categories.

The model cannot accept images or PDFs (without them being converted to images first) of any shape and requires that they be resized to a NumPy array of 128x128 with 3 channels for RGB. In this, each number represents a pixel brightness value ranging from 0.0-1.0. Note that image is in a NumPy array of float32 format with each pixel value normalised by dividing by 255.0 (max RGB value).

This model has not been designed to process Black and White images. It will still accept them if they have been converted to a 3-channel image as with RGB, however this is out of the scope of the model and while doing so should work, it is with the possibility of suboptimal performance in terms of accuracy.

The model may also falter if an image has multiple identifiable categories. This could occur if an image input into the program is a scan of a medical image beside a medical document, or if the image is a smartphone captured image of, say, a medical image such as an x-ray being held up against the background of a clinical setting or a patient. These are a few examples that could occur but had never been encountered at the time of creation of the model.

## How to Operate the Model
In order to get the model to function, the image data must be prepped as follows (Note: Any PDFs must be converted into an image before proceeding):

### Data Prep

1.	Import the Image in RGB form
2.	Resize the image to 128x128 pixels
3.	Convert the image into a NumPy Array of shape (128, 128, 3)
4.	Normalise the array by dividing by 255.0 (Max RGB value). This will also convert the data to float32 type.

### Running prediction on prepped Image Data
1.	Feed the prepped image into the model
2.	Process the probability array outputted by the model to get the value of the highest probability
3.	Determine the classification as Medical Image, Medical Document or Clinical Photograph (in this specific order) by referring to the processed probability array

 
### One Function to do it all

Data Prep and Model Running can be performed in one easy to use function that takes in an image path and outputs the class name as text. Code is as follows:

def predict_class(path_image, path_model, image_size=128):
   # Load and Resize Image
  img = cv2.cvtColor(cv2.imread(path_image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
  img_transformed = cv2.resize(img, (image_size, image_size), interpolation = cv2.INTER_CUBIC)
  # Convert to numpy array
  img_transformed = np.array(img_transformed).reshape(-1, image_size, image_size, 3)
   # Normalise the array and convert to float32
  img_transformed = img_transformed/255.0

  # Load Model
  model = tf.keras.models.load_model(path_model)
   # Predict Class of Image
  prediction_array = model.predict(img_transformed)
   # Output the Prediction Array as Class Name
  prediction_array = np.reshape(prediction_array, -1)
  prediction_array_index = np.where(prediction_array == np.amax(prediction_array))
   if prediction_array_index[0] == 0:
      prediction_class = 'Image Type: Medical Image'
  elif prediction_array_index[0] == 1:
      prediction_class = 'Image Type: Medical Document'
  elif prediction_array_index[0] == 2:
      prediction_class = 'Image Type: Clinical Photo'
    
  plt.imshow(img)
  plt.show()
    
  print(prediction_class)

## Contact Details

Mudit Dalmia
mdalmia95+career@gmail.com
+91 98704 55977
