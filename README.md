<div id="top"></div>

<!-- PROJECT NAME-->
<br />
<div align="center">
  <h2 align="center">Performance Comparison of CNN Models for Chest X-Ray Image Classification to Detect Covid 19</h2>
</div>

<br>
<br>

<!-- TABLE OF CONTENTS -->
<details>
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
        <li><a href="#download-dataset">Download Dataset</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#running-the-project">Running the Project</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<br>
<br>


<!-- ABOUT THE PROJECT -->
## About The Project

In this project, we design a deep learning system to extract features and detect COVID-19 from chest X-ray images. Analysis of lung diseases can be started from the x-ray images of the chest as it provides general information for initial inquiry. This information helps determine covid-19 and pneumonia which results in affecting the lungs. We are building our system <b>using six Convolution Neural Network models, simple CNN, LeNet, AlexNet, DenseNet121, ResNet50 and VGG16 to classify chest x-ray images (normal, covid, pneumonia)</b>. The performance of each of them is compared using tensorflow platform in python to find the best model to detect COVID-19. Deep learning can extract characteristics more accurately and automatically than previous artificial intelligence approaches.As a result, deep learning may be used in a variety of situations. We are comparing six CNN models based on five performance criterias, accuracy, precision, f1 score, loss  and specificity. The results show that DenseNet121 shows less value for loss i.e. 0.125. The accuracy, precision, F1 score and specificity for DenseNet121 is 95.5,0.95,0.95,66 respectively. This shows that DenseNet121 CNN performed better than the other CNN models and successfully classified chest x-ray images.


<p align="right">(<a href="#top">back to top</a>)</p>



### Built With

Following libraries are being used in this project:

* [Python](https://www.python.org/)
* [Tensorflow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)
* [Sklearn](https://scikit-learn.org/stable/)
* [Seaborn](https://seaborn.pydata.org/)
* [Matplotlib](https://matplotlib.org/)
* [Numpy](https://numpy.org/)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

Following steps are required before running the project.

### Download Dataset

The dataset used in this project can be downloaded from Kaggle website on the link given below and placed in Data folder.
* Download Dataset [Chest X-ray (Covid-19 & Pneumonia)](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia)

### Installation

Following installations are required to run this project.

1. Download [Python 3.9](https://www.python.org/downloads/)

2. Install following packages
   ```sh
   py -m pip --version
   pip install --upgrade tensorflow
   pip install keras
   pip install scikit-learn
   pip install seaborn
   pip install matplotlib
   pip install numpy==1.21.5
   ```

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- Running the Project-->
## Running the Project

Following steps are required to successfully run the project files.

1. Following variables set the directories path in 7 coding files, under the _Code Files_ folder.

   ```sh
   TRAIN_PATH = "Data/train"
   TEST_PATH = "Data/test"

   TRAIN_COVID_PATH = "Data/train/COVID19"
   TRAIN_NORMAL_PATH = "Data/train/NORMAL"
   TRAIN_PNE_PATH = "Data/train/PNEUMONIA"

   VAL_NORMAL_PATH = "Data/test/COVID19"
   VAL_PNEU_PATH = "Data/test/NORMAL"
   VAL_COVID_PATH = "Data/test/PNEUMONIA"
   ```
2. Once the directories path is set, run the following files one by one.
   ```sh
   Code Files/Exploratory_Analysis.py
   Code Files/CNN.py
   Code Files/LENET.py
   Code Files/ALEXNET.py
   Code Files/DENSENET121.py
   Code Files/RESNET50.py
   Code Files/VGG16.py
   ```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

* Agha Ahmad - agha.ahmad@ryerson.ca

* Hina Awan - hina.awan@ryerson.ca

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- Acknowledgments -->
## Acknowledgments

* [Deep learning based detection of COVID-19 from chest X-ray images](https://doi-org.ezproxy.lib.ryerson.ca/10.1007/s11042-021-11192-5)

* [Automatic COVID-19 detection from X-ray images using ensemble learning with convolutional neural network](https://doi-org.ezproxy.lib.ryerson.ca/10.1007/s10044-021-00970-4)

* [A non-invasive methodology for the grade identification of astrocytoma using image processing and artificial intelligence techniques](https://scholar.google.com/scholar_lookup?journal=Expert+Systems+with+Applications&title=A+non-invasive+methodology+for+the+grade+identification+of+astrocytoma+using+image+processing+and+artificial+intelligence+techniques&author=M.M.+Subashini&author=S.K.+Sahoo&author=V.+Sunil&author=S.D.+Easwaran&volume=43&publication_year=2016&pages=186-196&)

* [Automatic detection from X-ray images utilizing transfer learning with convolutional neural networks](https://arxiv.org/pdf/2003.11617.pdf)


<p align="right">(<a href="#top">back to top</a>)</p>

