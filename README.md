<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** roberttovornik, imgbiom-ear-detector, twitter_handle, robert.tovorni, project_title, project_description
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!-- [![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url] -->



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <!-- <a href="https://github.com/roberttovornik/imgbiom-ear-recognition">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->

  <h3 align="center"> Ear recognition in attempt of person identification</h3>

  <p align="center">
    Fine tuning of MobileNet_v2, EfficientDet, ResnetInception_v2 and Few-shot/One-shot learning with Keras reptile for the task of person identification.
    <br />
    <!-- <a href="https://github.com/roberttovornik/imgbiom-ear-recognition"><strong>Explore the docs »</strong></a> -->
    <!-- <br /> -->
    <!-- <br /> -->
    <a href="https://github.com/roberttovornik/imgbiom-ear-recognition/tree/main/test/output">View Demo</a>
    ·
    <a href="https://github.com/roberttovornik/imgbiom-ear-recognition/issues">Report Bug</a>
    ·
    <a href="https://github.com/roberttovornik/imgbiom-ear-recognition/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
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
        <li><a href="#dependencies">Dependencies</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#evaluation">Evaluation (optional)</a></li>
    <li><a href="#Training">Training</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->
An implementation of SingleShotDetector (SSD) was used for the task of ear detection, with classification of either left or right ear. Base [SSD](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz) model found at [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection), was fine-tuned on a a custom [dataset](http://awe.fri.uni-lj.si/). for the final prediction. Evaluation was made. The exported model can be found [here](https://github.com/roberttovornik/imgbiom-ear-recognition/tree/main/training/exported_models).

![Ear recognition demo](test/output/output-efd-th-1/the-umbrella-academy-season-2-poster--a3d3be0.png?raw=true)

### Built With

* [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
* [Keras Few-Shot learning with Reptile](https://keras.io/examples/vision/reptile/)
* [Ear Biometric Database in the Wild](http://awe.fri.uni-lj.si/)



<!-- GETTING STARTED -->
## Getting Started

### Dependencies

This is an example of how to list things you need to use the software and how to install them.
* Python 3.7+
* Tensorflow 2
* Opencv 4.0+
* PIL
* (optional) pascalvoc writer, labelimg

Tensorflow requirements:
* CUDA 10.1 (used V10.1.243)
* cuDNN 7.6.5

### Installation

1. Clone the repo (or download the zip)
   ```sh
   git clone https://github.com/roberttovornik/imgbiom-ear-recognition.git
   ```
2. Setup conda environment
   ```sh
   conda env create -f envs/environment.yml
   ```
   or pip
   ```sh
   pip install -r requirements.txt
   ```



<!-- USAGE EXAMPLES -->
## Usage

Activate conda environment  (default_name: **biom-task3**)
```sh
   conda activate environment_name
   ```

Copy images into folder "**test/images**". Some demo images are provided.

Run the script
```sh
   python code/detection_demo.py
   ```
Output of the detection will be (default) **saved** into **test/output**.


<!-- _For more examples, please refer to the [Documentation](https://example.com)_ -->

### Dataset
Either collect the [Ear Biometric Database in the Wild](http://awe.fri.uni-lj.si/) dataset or select a custom dataset for different detection problem.

If the dataset is not annotated, meaning you only have some images, you can manually do so using this simple tool: [labelImg](https://github.com/tzutalin/labelImg.git)
Make sure to save in pascal-voc format (output is .xml for each image).

Once complete, split the data into train/test set (80-20) and move them into training/images/train  and training/images/test respectively.

(*xml_to_csv, generate_tf_records)

##### Dataset A (AWEForSegmentation)
Images of person close-ups (such as test images). Useful for object detection training.
##### Dataset B (AWEDataset)
Extracted / cropped images of objects (ears). Should be split into subdirectories: train-test-val(optional) with each class containing own images.
For custom datasets organize as such:

##### CUSTOM dataset
```
images/
    train/
        object-1/*.jpg or *.png
        object-2/
        ...
        object-n/
    validate/
        object-1/*.jpg or *.png
        object-2/
        ...
        object-n/
    test/
        object-1/*.jpg or *.png
        object-2/
        ...
        object-n/
```


<!-- TRAINING EXAMPLES -->
## Training
Main training steps
* Dataset collection
* (if required) Annotate the dataset
* Install Tensorflow object detection API
* Download pretrained model
* Update configuration
* Start the TRAINING process
Notice: Most of the training can be done through provided .ipynb in **code/notebooks** or **code/colab**. If you don't have GPU you can import either to google colab.
Make sure to updated the dataset paths and labels as required.

### Pretrained models
Download one of the pretrained models from the [Tensorflow 2 Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). The model used for the training of ear recognition were:
* [SSD MobileNet V2 FPNLite 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz)
* [EfficientDet D1 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz)
* [Faster R-CNN Inception ResNet V2 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.tar.gz).

Place the extracted model into "**training/pretrained-models/**.."
### Update configuration files
Make sure to select proper "label.map" depending on problem: human-ear detection(1), person-identification(100), person-id+ear recognition (200) 
* label_map.pbtxt
* pipeline config (batch, path_to_models)
* ..


## Evaluation
* Run evaluation export .zip file in **training/exported_models**
* Fix paths in **training/annotations** and paths to images and annotations in **training/models/../config**. 
* Place images inside the **test/images** 
* Run one of the scripts in **code/scripts** names "..demo.."
* Or run the code inside one of the jupyter notebooks

### Additional packages setup
Install tensorflow object detection API, following the instructions [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md). Git clone the the tensorflow API into "models" directory. When running installation processes make sure to have te right conda (or pip) environment activated.

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

<!-- Name Surname - [@twitter_handle](https://twitter.com/twitter_handle) - email -->
Robert Tovornik - robert.tovornik@gmail.com
Project Link: [https://github.com/roberttovornik/imgbiom-ear-recognition](https://github.com/roberttovornik/imgbiom-ear-recognition)



<!-- ACKNOWLEDGEMENTS -->
<!-- ## Acknowledgements

* []()
* []()
* []() -->





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/roberttovornik/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/roberttovornik/repo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/roberttovornik/repo.svg?style=for-the-badge
[forks-url]: https://github.com/roberttovornik/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/roberttovornik/repo.svg?style=for-the-badge
[stars-url]: https://github.com/roberttovornik/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/roberttovornik/repo.svg?style=for-the-badge
[issues-url]: https://github.com/roberttovornik/repo/issues
[license-shield]: https://img.shields.io/github/license/roberttovornik/repo.svg?style=for-the-badge
[license-url]: https://github.com/roberttovornik/repo/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/roberttovornik
