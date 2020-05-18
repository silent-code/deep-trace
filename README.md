

[image_1]: ./images/ acc_loss_from_scratch_adam.png
[image_2]: ./images/ tsne_projection.png
[image_3]: ./images/ cct_embedding2.png
[image_4]: ./images/ roc_curve_Infected.png
[image_5]: ./images/ roc_curve_Exposed.png
[image_6]: ./images/ roc_curve_Susceptible.png
[image_7]: ./images/ confusion_matrix_Susceptible.png

# Deep-learning framework for cnn-autoencoder-based classification of wqx track plots

Deep-trace is a graph-based deep-learning neural net for contact tracing. Conventional methods can only exploit knowledge of a person&#39;s contacts, essentially a graph with nodes representing people and edges connecting contact between people. The proposed method allows us to utilize information stored in the graph contacts as well as node features. In this particular case we use the covid vulnerability index to assign a feature vector to each node. We are then able to learn the contact network based not only on the graph node and edgelist specification, but also the vulnerability feature mapping. Thus we create a three-dimensional node embedding for new contacts that shows an assessment of their likelihood of being in one of three exposure categories – Infected, Exposed or Susceptible. This low dimensional embedding allows contact tracing personnel to prioritize who they should contact and have test in cases where a pandemic is evolving too quickly under limited resources. One can quickly identify and prioritize which persons to contact and isolate.
<br />

This framework demonstrates:<br />
- The use of combined cnn with autoencoder to classifiy uuv tracks  <br />
- Unsupervised training using only the inlier class (uuv tracks)  <br />
- Quality curated dataset for training <br />

Further work: <br />
- Extend inlier class to include uuv lawn mower turn segments (right now we only include straight path segments)
- Hyperparameter tuning
- Analyze augmenting existing WQX SVM classifier with deep_track_ae classifications
- Increase input image size for training to improve accuracy
- Use k-folds averaging to increase ROC performance estimate reliability

The repository contains:

* Code for training (train.py) and testing performance (evaluate_performance.py) on wqx uuv track data
* UUV track plot data is here: \\ead-fs1\Huge\Users\josseran\wqx_ml_deep_track_data

## Installation

### Clone this repo

git clone gitlab@repositories.arlut.utexas.edu:wqx_ml/wqx_deep_track.git

### Install virtualenv and opencv-contrib-python 

#### Various OS:

See [here](https://www.pyimagesearch.com/opencv-tutorials-resources-guides/): 

#### For Mac OS simply pip install opencv-contrib-python

See [here](https://www.pyimagesearch.com/2018/09/19/pip-install-opencv/)

### Install requirements

pip install -r requirements.txt


## Usage

### Training:

python train_unsupervised_autoencoder.py --dataset datasets/data_augmented3 --test_data_dir datasets/test_data_aug3 --model output/autoencoder_aug3.model --epochs 100


### Evaluation:

python evaluate_performance.py --dataset_dir datasets/test_data --model output/autoencoder.model --max_alert_rate 4

## Method:

Convolutional autoencoder flow:

![alt text][image_6]

Autoencoder anomaly detection is performed by calculating the mean-square-error between the input and reconstructed track plots. The upper flow diagram in the following image shows the 
mse is high when the autoencoder is fed a track which deviates from a straight track (lower flow diagram):

![alt text][image_5]

Figure below shows a TSNE projection of the data onto three dimensions for a simulated case study of 27 infected, 519 susceptible, and 419 exposed individuals:

![alt text][image_2]





## ROC Analysis:

The following ROC curve shows the performance on test data for the quality curated dataset in ./datasets/data_augmented3. Quality
curation entails culling bad examples from the data to improve the quality of the training and validation sets and then augmenting
this improved data to increase the data set size.

![alt text][image_3]

The next ROC curve shows the performance on test data for the original dataset in ./datasets/data. 

![alt text][image_4]

## Critical Dependencies:

* Python3
* Tensorflow  >= 2.0
* Keras > 2.3
* Opencv > 4.1.0

## Training 

Training errors on augmented3 dataset:

![alt text][image_1]

### Dataset:

Dataset track plot collage shows raw track plot input images for uuv tracks (straight line segments) and ambient tracks:

![alt text][image_7]


107 Targets<br />
214 Non-targets<br />
Tracker speed is filtered at 3 kn<br />
Download data from: \\ead-fs1\Huge\Users\josseran\wqx_ml_deep_track_data

### Data directory structure:

 |-datasets<br />
 |---data<br />
 |-----non_target<br />
 |-----target<br />
 |---data_augmented<br />
 |-----non_target<br />
 |-----target<br />
 |---data_augmented2<br />
 |-----non_target<br />
 |-----target<br />
 |---data_augmented3<br />
 |-----non_target<br />
 |-----target<br />
 


### Data Augmentations
The following data augmentation has been applied to increase the no of images in the training set:

-Flip horizontal<br />
-Lighting<br />
-Zooming<br />
-Warping<br />


## References

***

Tutorial on Keras autoencoders: https://www.pyimagesearch.com/2020/03/02/anomaly-detection-with-keras-tensorflow-and-deep-learning/

Configuring Virtual Environment for Deep Learning: https://www.pyimagesearch.com/2018/09/19/pip-install-opencv/

