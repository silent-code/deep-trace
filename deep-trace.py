import networkx as nx
import pandas as pd
import os
import numpy as np
from time import time
from utils.plot_data import PlotData
import matplotlib.pyplot as plt
import stellargraph as sg
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE
# note that using "from keras" will not work in tensorflow >= 2.0
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, \
    classification_report, accuracy_score

def plot_history(history):
    metrics = sorted(history.history.keys())
    metrics = metrics[:len(metrics) // 2]
    for m in metrics:
        plt.plot(history.history[m])
        plt.plot(history.history['val_' + m])
        plt.title(m)
        plt.ylabel(m)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()

# Set the graph edgelist with the Cora target-source citation data
data_dir = os.path.expanduser("./datasets/contacts")
cora_location = os.path.expanduser(os.path.join(data_dir, "cct.contacts"))
edgelist = pd.read_csv(os.path.join(data_dir, "cct.contacts"), sep='\t', header=None, names=["target", "source"])
edgelist["label"] = "contacts"
g_nx = nx.from_pandas_edgelist(edgelist[0:1348], 'target', 'source')

# Set each node attribute as the 'subject' data in the last field of the cora feature file
cct_feature_dir = os.path.expanduser("./datasets/covid_vulnerability_features")
cct_features_location = os.path.expanduser(os.path.join(cct_feature_dir, "cct_features.csv"))
node_attr = pd.read_csv(cct_features_location)
node_attr = node_attr.drop(columns='personId')
node_attr.loc[node_attr['Gender'] == 'male', 'Gender'] = 1.
node_attr.loc[node_attr['Gender'] == 'female', 'Gender'] = 0.
node_attr['Age'] = node_attr['Age'].div(max(node_attr['Age']))
node_attr = node_attr.replace([True, False], [1., 0.])
feature_names = node_attr.columns.values.tolist()
# Let's artificially create SEI classification labels:
# S = susceptible
# E = exposed
# I = infected case
f1 = 'Diagnosis of Lung disease due to external agents in the previous 12 months'
f2 = 'Diagnosis of Influenza in the previous 12 months'
f3 = 'Diagnosis of Pneumothorax in the previous 12 months'
f4 = 'Age'
f5 = 'Gender'
# Note: race would be a good feature component also!
# Tweak threshold for ~50% exposed rate
node_attr['subject'] = np.where(node_attr[f1] + node_attr[f2] + node_attr[f3] + node_attr[f5] + node_attr[f4] >= 1.2, 'E', 'S')
node_attr.loc[node_attr[f1] + node_attr[f2] + node_attr[f3] + node_attr[f5] + node_attr[f4] >= 2, 'subject'] = 'I'
print('Number infected:', len(np.where(node_attr['subject'] == 'I')[0])/1.*1.)
print('Number exposed:', len(np.where(node_attr['subject'] == 'E')[0])/1.*1.)
print('Number susceptible:', len(np.where(node_attr['subject'] == 'S')[0])/1.*1.)
values = {str(row.tolist()[0]): row.tolist()[-1] for _, row in node_attr.iterrows()}
nx.set_node_attributes(g_nx, values, 'subject')
node_attr.index = [*g_nx.nodes]

# Select the largest connected component. For clarity we ignore isolated
# nodes and subgraphs; having these in the data does not prevent the
# algorithm from running and producing valid results.
g_nx_ccs = (g_nx.subgraph(c).copy() for c in nx.connected_components(g_nx))
g_nx = max(g_nx_ccs, key=len)
print("Largest subgraph statistics: {} nodes, {} edges".format(
    g_nx.number_of_nodes(), g_nx.number_of_edges()))

# Get node feature data for only the nodes remaining largest connected component graph
node_data = node_attr
node_data = node_data[node_data.index.isin(list(g_nx.nodes()))]

# Train a graph-ML model that will predict the "subject" attribute on the nodes. These subjects are one of 3 categories
set(node_data["subject"])

# For machine learning we want to take a subset of the nodes for training,
# and use the rest for testing. We'll use scikit-learn again to do this
# These are train / test dataframe splits
train_data, test_data = model_selection.train_test_split(node_data, train_size=0.6, test_size=None,
                                                         stratify=node_data['subject'], random_state=42)
# Create dataframe to ndarray transform
target_encoding = feature_extraction.DictVectorizer(sparse=False)
# Create the training and test split label one-hot-encoding labels for the generators below
# This means, that the DictVectorizer needs to be fitted prior to
# transforming target_encoding into it's corresponding matrix format.
# You need to call vec.fit(target_encoding) followed by vec.transform(target_encoding),
# or more succintly X_train = vec.fit_transform(target_encoding).
# DictVectorizer needs to know the keys of all the passed dictionaries,
# so that the transformation of unseen data consistently yields the same number of columns and column order.
train_targets = target_encoding.fit_transform(train_data[["subject"]].to_dict('records'))
test_targets = target_encoding.fit_transform(test_data[["subject"]].to_dict('records'))
# Get just the node feature vectors from the node dataframe
node_features = node_data[feature_names]
# print(node_features.head(2))
# Now create a StellarGraph object
G = sg.StellarGraph.from_networkx(g_nx, node_features=node_features)
# Print help for GraphSAGENodeGenerator
# help(GraphSAGENodeGenerator)
batch_size = 50
num_samples = [10, 20, 10]
generator = GraphSAGENodeGenerator(G, batch_size, num_samples)
# Create a test data generator given training set node index and it's label
train_gen = generator.flow(train_data.index, train_targets)

graphsage_model = GraphSAGE(
    layer_sizes=[64, 64, 64],
    generator=generator,
    bias=True,
    dropout=0.5,
)

# Create the network model using the graph input and output tensors and a softmax prediction layer
x_inp, x_out = graphsage_model.in_out_tensors()
prediction = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

# Cost function
model_loss_function = losses.categorical_crossentropy

# Training metric
model_metrics = ["acc"]

# define optimizers
model_optimizer_rmsprop0 = 'rmsprop'
model_optimizer_adam0 = 'adam'
model_optimizer_adam = optimizers.Adam(lr=0.005)
model_optimizer_sgd = optimizers.SGD(lr=1e-5, decay=1e-6, momentum=0.45, nesterov=True)
model_optimizer_rmsprop = optimizers.RMSprop(lr=1e-5, decay=0.9, momentum=0.5, epsilon=1e-10, centered=True)
model_optimizer_rmsprop1 = optimizers.RMSprop(lr=1e-5, decay=1e-6, momentum=0.49,  centered=True)
use_best_weights = False
# Use this after getting best weights
if use_best_weights:
    model_optimizer = model_optimizer_rmsprop1
else:
    # Use this as first run
    model_optimizer = model_optimizer_adam

# Create the network model
model = Model(inputs=x_inp, outputs=prediction)
print(model.summary())
#Epoch 00016: val_loss improved from 0.40059 to 0.38344, saving model to .\trained_for_pred\cct_graphsage_node_inference\model\log\Best-weights-my_model-016-0.4879-0.7858.h5
#12/12 - 5s - loss: 0.4869 - acc: 0.7858 - val_loss: 0.3834 - val_acc: 0.8756
#Epoch 17/50
#Epoch 00008: val_loss improved from 0.38307 to 0.38293, saving model to .\trained_for_pred\cct_graphsage_node_inference\model\log\Best-weights-my_model-008-0.4605-0.8014.h5
#12/12 - 5s - loss: 0.4658 - acc: 0.8014 - val_loss: 0.3829 - val_acc: 0.8731
#Epoch 9/10
if use_best_weights:
    # Load the best weights and compile the model
    best_weights = '.\\trained_for_pred\\cct_graphsage_node_inference\\model\\log\\Best-weights-my_model-008-0.4605-0.8014.h5'
    model.load_weights(best_weights)
model.compile(
    optimizer=model_optimizer,
    loss=model_loss_function,
    metrics=model_metrics,
)

# Create a test data generator given test node index and it's label
test_gen = generator.flow(test_data.index, test_targets)

# Prepare callbacks
model_type = 'cct_graphsage_node_inference'
train_log_path = '.\\trained_for_pred\\' + \
    model_type + '\\model\\log\\model_train.csv'
train_checkpoint_path = '.\\trained_for_pred\\' + model_type + \
    '\\model\\log\\Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.h5'
model_tensorboard_log = '.\\training_log\\tensorboard\\'
csv_log = callbacks.CSVLogger(train_log_path, separator=',', append=False)
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
checkpoint = callbacks.ModelCheckpoint(
    train_checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(
    log_dir=model_tensorboard_log + "{}".format(time()))
callbacks_list = [csv_log, tensorboard, checkpoint]

# Train the network
history = model.fit(
    train_gen,
    epochs=50,
    validation_data=test_gen,
    verbose=2,
    callbacks=callbacks_list,
    shuffle=True,
)

plot_history(history)

test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))
######################################################################################################################
# start the evaluate and report process
print("===================== starting evaluation and report generation =========================")
# Create plotter object
plotter = PlotData()
# paths to save outputs
save_plt_cm = './trained_for_pred/' + model_type + '/stats/confusion_matrix.png'
save_plt_normalized_cm = './trained_for_pred/' + \
    model_type + '/stats/confusion_matrix_normalized.png'
save_plt_roc = './trained_for_pred/' + \
    model_type + '/stats/roc_curve.png'
save_eval_report = './trained_for_pred/' + \
    model_type + '/stats/eval_report.txt'
save_plt_accuracy = './trained_for_pred/' + \
    model_type + '/stats/model_accuracy.png'
save_plt_loss = './trained_for_pred/' + \
    model_type + '/stats/model_loss.png'
save_plt_learning = './trained_for_pred/' + \
    model_type + '/stats/model_learning.eps'
train_log_data_path = './trained_for_pred/' + \
    model_type + '/model/log/model_train.csv'
# for confusion matrix plotting
classification_list = ["Exposed", "Infected", "Susceptible"]
# NOTE: Do not shuffle the test data! This is an issue with Keras.
# See: https://github.com/keras-team/keras/issues/4225
# and: https://github.com/keras-team/keras/issues/5558
Y_true = test_targets
Y_pred = model.predict(test_gen)
n_classes = Y_true.shape[1]
# Save multiclass confusion matrices and ROCs
for i in range(n_classes):
    # Confusion Matrix and Classification Report
    y_true = Y_true[:, i]
    y_pred = np.round(Y_pred[:, i])
    # plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_plot_labels = ["Other", classification_list[i]]
    save_plt_cm_to = save_plt_cm[:-4] + '_' + classification_list[i] + '.png'
    plotter.plot_confusion_matrix(
        cm, cm_plot_labels, save_plt_cm_to, title='Confusion Matrix')
    save_plt_normalized_cm_to = save_plt_normalized_cm[:-4] + '_' + classification_list[i] + '.png'
    plotter.plot_confusion_matrix(
        cm, cm_plot_labels, save_plt_normalized_cm_to, normalize=True, title='Normalized Confusion Matrix')
    # Compute ROC curve and ROC area for each class
    save_plt_roc_to = save_plt_roc[:-4] + '_' + classification_list[i] + '.png'
    roc_auc = plotter.plot_roc(y_true, Y_pred[:, i], save_plt_roc_to)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    print('mean absolute error: ' + str(mae))
    print('mean squared error: ' + str(mse))
    print('Area Under the Curve (AUC): ' + str(roc_auc))
    c_report = classification_report(
        y_true, y_pred, target_names=cm_plot_labels)
    print(c_report)
    save_eval_report_to = save_eval_report[:-4] + '_' + classification_list[i] + '.txt'
    #delete_file(save_eval_report)
    with open(save_eval_report_to, 'a') as f:
        f.write('\n\n')
        f.write('******************************************************\n')
        f.write('**************   Evalaluation Report   ***************\n')
        f.write('******************************************************\n')
        f.write('\n\n')
        f.write('- Accuracy Score: ' + str(accuracy))
        f.write('\n\n')

        f.write('- Mean Absolute Error (MAE): ' + str(mae))
        f.write('\n\n')

        f.write('- Mean Squared Error (MSE): ' + str(mse))
        f.write('\n\n')

        f.write('- Area Under the Curve (AUC): ' + str(roc_auc))
        f.write('\n\n')

        f.write('- Confusion Matrix:\n')
        f.write(str(cm))
        f.write('\n\n')

        f.write('- Normalized Confusion Matrix:\n')
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        f.write(str(cm))
        f.write('\n\n')

        f.write('- Classification report:\n')
        f.write(str(c_report))

        f.close()

train_validation = ['train', 'validation']
data = pd.read_csv(train_log_data_path)
acc = data['acc'].values
val_acc = data['val_acc'].values
loss = data['loss'].values
val_loss = data['val_loss'].values

# plot metrics to the stats dir
plotter.plot_2d(acc, val_acc, 'epoch', 'accuracy',
                 'Model Accuracy', train_validation, save_plt_accuracy)
plotter.plot_2d(loss, val_loss, 'epoch', 'loss',
                 'Model Loss', train_validation, save_plt_loss)
plotter.plot_model_bis(data, save_plt_learning)

#####################################################################################################################
# To get an idea of how the prediction errors are distributed visually
# load the graph in yEd Live and apply a radial layout:
#  get the predictions themselves for all nodes using another node iterator:
all_nodes = node_data.index
all_mapper = generator.flow(all_nodes)
all_predictions = model.predict(all_mapper)
# invert the one-hot encoding
node_predictions = target_encoding.inverse_transform(all_predictions)
results = pd.DataFrame(node_predictions, index=all_nodes).idxmax(axis=1)
df = pd.DataFrame({"Predicted": results, "True": node_data['subject']})
df.head(10)
# augment the graph with the true vs. predicted label for visualization purposes:
for nid, pred, true in zip(df.index, df["Predicted"], df["True"]):
    g_nx.nodes[nid]["subject"] = true
    g_nx.nodes[nid]["PREDICTED_subject"] = pred.split("=")[-1]
# add isTrain and isCorrect node attributes:
for nid in train_data.index:
    g_nx.nodes[nid]["isTrain"] = True
for nid in test_data.index:
    g_nx.nodes[nid]["isTrain"] = False
for nid in g_nx.nodes():
    g_nx.nodes[nid]["isCorrect"] = g_nx.nodes[nid]["subject"] == g_nx.nodes[nid]["PREDICTED_subject"]
# Save the graphml object
pred_fname = "pred_n={}.graphml".format(num_samples)
nx.write_graphml(g_nx,'./nodepredictions.graphml')

######################################################################################################################
# Node embeddings
# Evaluate node embeddings as activations of the output of graphsage layer stack, and visualise them,
# coloring nodes by their subject label.
# The GraphSAGE embeddings are the output of the GraphSAGE layers, namely the x_out variable.
# Let’s create a new model with the same inputs as we used previously x_inp but now the output is
# the embeddings rather than the predicted class. Additionally note that the weights trained previously
# are kept in the new model.
embedding_model = Model(inputs=x_inp, outputs=x_out)

emb = embedding_model.predict(all_mapper)
# This is the number of nodes in the largest connected subgraph x outputa layer size
print('Embedding shape:', emb.shape)


from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches

X = emb
y = np.argmax(target_encoding.transform(node_data[["subject"]].to_dict('records')), axis=1)
n_components = 3
if X.shape[1] > 2:
    transform = TSNE
    trans = transform(n_components=n_components)
    emb_transformed = pd.DataFrame(trans.fit_transform(X), index=node_data.index)
    emb_transformed['label'] = y
else:
    emb_transformed = pd.DataFrame(X, index=node_data.index)
    emb_transformed = emb_transformed.rename(columns={'0': 0, '1': 1})
    emb_transformed['label'] = y

alpha = 0.7
color_list = ['b', 'r', 'g']
colors = [color_list[i] for i in emb_transformed['label']]
classification_list = ["Exposed", "Infected", "Susceptible"]
classifications = [classification_list[i] for i in emb_transformed['label']]
# Make sure classification list is in the right order:
print('Number Infected:', classifications.count('Infected'))
print('Number Susceptible:', classifications.count('Susceptible'))
print('Number Exposed:', classifications.count('Exposed'))

if n_components == 2:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(emb_transformed[0], emb_transformed[1], c=colors, alpha=alpha)
    ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
else:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(emb_transformed[0], emb_transformed[1], emb_transformed[2], c=colors, alpha=alpha)
recs = []
for i, _  in enumerate(color_list):
    recs.append(mpatches.Rectangle((0,0),1,1,fc=color_list[i]))

plt.title('{} visualization of GraphSAGE embeddings for CCT dataset'.format(transform.__name__))
ax.legend(recs, classification_list, loc=4)
ax.grid(True)
plt.show()
