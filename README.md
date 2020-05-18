

[image_1]: ./images/acc_loss_from_scratch_adam.png
[image_2]: ./images/tsne_projection.png
[image_3]: ./images/cct_embedding2.png
[image_4]: ./images/roc_curve_Infected.png
[image_5]: ./images/roc_curve_Exposed.png
[image_6]: ./images/roc_curve_Susceptible.png
[image_7]: ./images/confusion_matrix_Susceptible.png

# Graph-based deep-learning neural net for contact tracing

Deep-trace is a graphsage-based deep-learning neural net for contact tracing. Conventional methods can only exploit knowledge of a person's contacts which is essentially a graph with nodes representing people and edges connecting contact between people. The proposed method allows us to utilize information stored in the graph contacts as well as node features. In this particular case we use the covid vulnerability index to assign a feature vector to each node. We are then able to learn the contact network based not only on the graph node and edgelist specification, but also the vulnerability feature mapping. Thus we create a three-dimensional node embedding for new contacts that shows an assessment of their likelihood of being in one of three exposure categories – Infected, Exposed or Susceptible. This low dimensional embedding allows contact tracing personnel to prioritize who they should contact and have test in cases where a pandemic is evolving too quickly under limited resources. One can quickly identify and prioritize which persons to contact and isolate.
<br />

Figure below shows a TSNE projection of the data onto three dimensions for a simulated case study of 27 infected, 519 susceptible, and 419 exposed individuals:

![alt text][image_2]

This is a 2-D projection of the same TSNE embedding:

![alt text][image_3]



## Performance Analysis:

The following ROC curve shows the performance on test data for the infected, exposed and susceptible test classes, respectively:

![alt text][image_4]

![alt text][image_5]

![alt text][image_6]
## Critical Dependencies:

* Graphsage
* Python3
* Tensorflow  >= 2.0
* Keras > 2.3
* Opencv > 4.1.0

## Training 

Accuracy and loss plots for the training dataset during the initial from scratch training:

![alt text][image_1]

### Dataset:

The dataset consists of fictional contacts using the Cora dataset link data and the Covid19 vulnerability example feature data found here: https://github.com/closedloop-ai/cv19index




## References

***

put refs here

