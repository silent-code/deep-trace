

[image_1]: ./images/ acc_loss_from_scratch_adam.png
[image_2]: ./images/ tsne_projection.png
[image_3]: ./images/ cct_embedding2.png
[image_4]: ./images/ roc_curve_Infected.png
[image_5]: ./images/ roc_curve_Exposed.png
[image_6]: ./images/ roc_curve_Susceptible.png
[image_7]: ./images/ confusion_matrix_Susceptible.png

# Graph-based deep-learning neural net for contact tracing

Deep-trace is a graph-based deep-learning neural net for contact tracing. Conventional methods can only exploit knowledge of a person&#39;s contacts, essentially a graph with nodes representing people and edges connecting contact between people. The proposed method allows us to utilize information stored in the graph contacts as well as node features. In this particular case we use the covid vulnerability index to assign a feature vector to each node. We are then able to learn the contact network based not only on the graph node and edgelist specification, but also the vulnerability feature mapping. Thus we create a three-dimensional node embedding for new contacts that shows an assessment of their likelihood of being in one of three exposure categories – Infected, Exposed or Susceptible. This low dimensional embedding allows contact tracing personnel to prioritize who they should contact and have test in cases where a pandemic is evolving too quickly under limited resources. One can quickly identify and prioritize which persons to contact and isolate.
<br />

This framework demonstrates:<br />
- The use of graph convnet  <br />
- Semisupervised node training  <br />
- Quality curated dataset for training <br />

Further work: <br />
- Extend inlier class to include 
- Hyperparameter tuning
- Analyze augmenting existing data
- Increase input image size for training to improve accuracy
- Use k-folds averaging to increase ROC performance estimate reliability

The repository contains:

* Code for training (train.py) and testing performance (evaluate_performance.py) on 
* etc something

ed a track which deviates from a straight track (lower flow diagram):

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

put refs here

