<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
        <ul>
        <li><a href="#train">Train</a></li>
        <li><a href="#test">Test</a></li>
        <li><a href="#augmentationtesting">AugmentationTesting</a></li>
      </ul>
    <li><a href="#license">License</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
## Image-Classification-Python-Framework

The objective of image classification is to identify and portray, as a unique gray level (or color), the features occurring in an image in terms of the object or type of land cover these features actually represent on the ground. Image classification is perhaps the most important part of digital image analysis. This project is an end to end image classification framework, based on KERAS and TENSORFLOW.

Here's why:
* Your time should be focused on creating something amazing. A project that solves a problem and helps others
* You shouldn't be doing the same tasks over and over again

### Built With

* [Keras](https://keras.io/)
* [Tensorflow](https://www.tensorflow.org/)
* [Albumentations](https://albumentations.ai/)
* [Opencv](https://opencv.org/)



<!-- GETTING STARTED -->
## Getting Started

The project uses config files to work.The next steps will prevail how to create a so called config file that works for this framework.

<!-- USAGE EXAMPLES -->
## Usage

### Train
The script has 2 parameters:
- `--training_configuration_file` or `-config`    [string] Represents the path of the training configuration file (must be JSON format).
- `--checkpoint_path` or `-checkpoint`    [string] The path of the checkpoint file (must be h5 format).


The train script can be called by ussing a certain command
##### Example

- `image_classif_train` `-config` "C:/configs/train.json" `-checkpoint`[optional] "Z:/cats_dogs_TrainingData/checkpoints/20210917-150603cp-009.h5"


### Image-classification train.json parameters

#### Main parameters
- `dataset_path`[string] The location of the dataset on whom the model will train 
- `workspace_path`[string] Location of the folder where the training information will be uploaded
- `model_path`[string] Path to the model.py file in which there is the model to be used in a required format.
- `network`[node] Network configuration
- `optimizer`[float] A node in which the parameters of the optimizer can be configured.
- `augmentations`[list of nodes] A list of nodes that represent multiple augmentations
- `epochs`[int] The number of passes of the entire training dataset the machine learning algorithm has completed.
- `metrics`[list of strings] 


### `network` PARAMETERS

- `network/input_shape`[node] - Contains information about width height and depth( number of channels)
- `network/input_format`[node] - Contains information about the channels and the data type of the pixels.
- `network/resize`[node] - 

#### 'input_shape' 

- `network/input_shape/width`[int] -The width of the input image in pixels 
- `network/input_shape/height`[int]-The height of the input image in pixels 
- `network/input_shape/depth`[int] -Number of channels 

##### Example
```
{
  "width": 224,
  "height": 224,
  "depth": 3
}
```
#### 'input_format' 

- `network/input_format/channels`[string] -Represents the color profile of the images. Possible color profiles are [RGB,BGR,GRAY]
- `network/input_format/data_type`[string]- The datatype of the pixels from the image . Possible data types are [UINT8,FLOAT]

##### Example
```
{
  "channels": "RGB",
  "data_type": "FLOAT"
}
```
#### 'resize' 
- `network/resize/method`[string] - The method used for resizing the images from their size to the input size. Resize methods are [CROP,STRECH,LETTERBOX]
- `network/resize/params`[node] - Used only if the chosen method is CROP.

##### Example
###### If the resize method is not CROP 
```
{
"method": "LETTERBOX",
}
```

#### 'params'
- `network/resize/params/tl_ratio` [float] - HHVVJKHVBKJV
- `network/resize/params/br_ratio`[float] - HJKVGFIKHYVCV
- `network/resize/params/resize_after_crop`[string][optional] - The resize method used after cropping the image 

###### If the resize method is CROP 
```
{
"method": "CROP",
"params": 
{
"tl_ratio": 0.1,
"br_ratio": 0.1,
"resize_after_crop":"LETTERBOX"
}
```

### `optimizer` PARAMETERS

- `optimizer/name`[string] - The name of the used optimizer, possible options [ADAM,SGD,ADADELTA,ADAGRAD,RMSPROP]
- `optimizer/lr`[node] - This node contains the actual losing rate value and the schedule type of it with it's necessary parameters.
- `optimizer/params`[node] - The parameters of the chosen optimizer. They can be found here [OPTIMIZERS](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers) .Also if the params node is missing or if some of the optimizer parameters are not specified, they will automatically get their base form.

#### 'lr' 

- `optimizer/lr/value`[float] - The initial value of the losing rate
- `optimizer/lr/schedule`[string] - The schedule of the losing rate. Possible ones are [PolynomialDecay,ExponentialDecay]
- `optimizer/lr/params`[node] - The parameters of the chosen schedule. They can be found here [SCHEDULE](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules) .Also if the params node is missing or if some of the schedule parameters are not specified, they will automatically get their base form.

#### EXAMPLE
```
{
"name": "SGD",
"lr": {
    "value": 0.01,
    "schedule": "PolynomialDecay",
    "params":{
            "decay_steps": 10000}
             },
"params": {
    "momentum":0.9
          }
```
### `augmentations` VALUES

Augmentations is a list composed of nodes and each node represents a single augmentation.
Each node conformation is as it follows

- `name`[string] - The name of the aug. Usable aug [horizontalflip,randomcrop,centercrop,rotate,randombrightnesscontrast,flip]
- `params`[node] - It's parameters, also if the params node is not specifed or if it doesn't contain all the parameters of the augmentation they will automatically get their base form based on the augmenattion name. Also the parameters can be found [here](https://albumentations.ai/docs/api_reference/augmentations/).

#### Example
```
[
   {
       "name":"horizontalFlip",
       "params":{
           "p":0.5
       }
   },
   {
       "name":"rotate",
       "params":{
           "limit":90,
           "p":0.7,
            "interpolation":1,
            "border_mode":4
       }
    },
    {
       "name":"flip"
    }

]
```

### Test

The script has 3 parameters:
- `--test_configuration_file` or `-config`    [string] Represents the path of the test configuration file (must be JSON format).
- `--checkpoint_path` or `-checkpoint`    [string] The path of the checkpoint file (must be h5 format).
- `--preprocess_testing_images` Preprocess the testing images with the config params'

The test script can be called by ussing a certain command
##### Example

- `image_classif_test` `-config` "C:/configs/test.json" `-checkpoint` "C:/Training_data/checkpoints/20210916-131359cp-010.h5"


### Image-classification test.json parameters

#### Main parameters
- `images_path`[string] The location of a folder that contains images to be tested or the path directly to a single image.
- `labels_path`[string] The path to a json file that contains details for each class. This file is auto-generated when training in the workspace folder, but it can also be self created.

###### Example of a file of this type
```
{
    "0": {
        "name": "cats",
        "uid": "class_000"
    },
    "1": {
        "name": "dogs",
        "uid": "class_001"
    }
}
```

- `results_path`[string] Location where the bad results of the test will be stored
- `model_path`[string] Path to the model.py file in which there is the model to be used in a required format.
- `top_k`[int] Accuracy parameter that represents a set of k answer items with the highest ranking score according to the given scoring function.
- `network`[node] Network configuration, it's exactly the same as the train network.

### AugmentationTesting
The script has 4 parameters:
- `----augmentation_pipeline_file` or `-config` [string] Represents the path of the train configuration file or a configuration file that consist only of the augmentation nod found in the train config file (must be JSON format).
- `----image_path` or `-img`    [string]  "The path of the image to be augmented or a folder that contains multiple images to be augmented."
- `--nr_of_steps` or `-steps`  [string] "Number of augmentations that will be made on the photo with the same augmentation pipeline"
- `--output_path` or `-out`  [string][optional] "The path where a folder named "aug_img" will be created. This folder will store the augmented images aswell as a json for each that contains data about the augmentations made on the photos. If this parameter is not present the folder will be created in the working workspace"

The augmentation testing script can be called by ussing a certain command
##### Example

- `image_classif_aug_test` `-config` "C:/configs/train.json" `-img` "C:/output_aug/to_be_aug" `-steps` "10" `out` "Z:/output_aug"

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.
