#basic 
from tqdm import tqdm

#PyTorch
from skorch import NeuralNetBinaryClassifier
from skorch.callbacks import EarlyStopping, Checkpoint, EpochScoring, WandbLogger

#scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold

#imbalance-learn
from imblearn.under_sampling import RandomUnderSampler

#word2vec
from gensim.models import Word2Vec

#weights and biases
import wandb

#private files
from setup import *
from embedder import *
from neural_net import *


def setup_dataset(dataset, edge_embedder_type, network_type, embedding_dimension, random_walk_length, embeddings=embeddings):

  #setup embedder
  if edge_embedder_type == 'hadamard':
    Embedder = HadamardEmbedderFast
  elif edge_embedder_type == 'average':
    Embedder = AverageEmbedderFast

  #setup network type
  if network_type == 'string':
    #get embeddings for string network
    network_type = 'string'
    model_dir = embeddings + f"/human_{network_type}_embedding_model_dim:{embedding_dimension}_len:{random_walk_length}"
    model = Word2Vec.load(model_dir)
    dataset.embedder = Embedder(keyed_vectors=model.wv)
    return dataset

  elif network_type == 'sequence_similarity':
    #get embedding for sequence similarity network
    network_type = 'ss'
    model_dir = embeddings + f"/human_{network_type}_embedding_model_dim:{embedding_dimension}_len:{random_walk_length}"
    model = Word2Vec.load(model_dir)
    dataset.embedder = Embedder(keyed_vectors=model.wv)
    return dataset

  elif network_type == 'string_&_sequence_similarity':
    #get embeddings for string network
    model_dir = embeddings + f"/human_string_embedding_model_dim:{embedding_dimension}_len:{random_walk_length}"
    model_string = Word2Vec.load(model_dir)

    #get embedding for sequence similarity network
    model_dir = embeddings + f"/human_ss_embedding_model_dim:{embedding_dimension}_len:{random_walk_length}"
    model_ss = Word2Vec.load(model_dir)
    dataset.embedder = EmbedderList([Embedder(keyed_vectors=model_string.wv), Embedder(keyed_vectors=model_ss.wv)])
    return dataset

  else:
    raise KeyError('network_type must be either string, sequence_similarity or  string_&_sequence_similarity')

def my_roc_auc(net, X, y):
    y_proba = net.predict_proba(X)[:, 1]
    return roc_auc_score(y, y_proba)

def my_average_precision(net, X, y):
    y_proba = net.predict_proba(X)[:, 1]
    return average_precision_score(y, y_proba)

def build_neural_network(input_dim, fc_layer_size, num_layers_hidden, network_type, dropout):
  if network_type == 'string_&_sequence_similarity':
    return NeuralNetwork(input_dim*2, fc_layer_size, num_layers_hidden, dropout)
  else:
    return NeuralNetwork(input_dim, fc_layer_size, num_layers_hidden, dropout)

def custom_split(train_dataset, train, val, sampling_strategy=1):
  #get training and validation subsets
  training_set = train_dataset.get_subset(train)
  training_set = resample_dataset(training_set, sampling_strategy=sampling_strategy)
  validation_set = train_dataset.get_subset(val)
  validation_set = resample_dataset(validation_set, sampling_strategy=1)
  print(f"The length of the validation set is {len(validation_set)}")
  print(f"The length of the training set is {len(training_set)}")
  return training_set, validation_set


def resample_dataset(dataset, sampling_strategy=1):
  rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
  edges_res, labels_res = rus.fit_resample(dataset.edges, dataset.labels)
  dataset_res = EdgesDataSet(edges_res, labels_res, embedder=dataset.embedder)
  return dataset_res

def build_optimizer(optimizer):
  if optimizer == 'sgd':
    return torch.optim.SGD
  elif optimizer == 'adam':
    return torch.optim.Adam
  else:
    return torch.optim.AdamW

def train_wb(config=None, train_dataset=None): 

  #setup weights and biases run
  wandb_run=wandb.init(config=config)
  config=wandb_run.config
  
  # #set the sweep name
  # wandb_run.config.update({"name": "_".join([f"{k}={v}" for k,v in config.items()])})
  # print(wandb_run.config.name)

  #setup dataset
  train_dataset = setup_dataset(train_dataset, config.edge_embedder_type, config.network_type, config.embedding_dimension, config.random_walk_length)

  #device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"The device is {device}")

  #pos_weight calculation
  pos_weight = torch.tensor(1/config.class_imbalance_undersampling)
  print(f"The weight of the positive class is: {pos_weight} while the weight of the negative class is 1.0")

  #indices for validation and training
  indices = [i for i in range(len(train_dataset))]
  train, val = train_test_split(indices, test_size=0.33, random_state=SEED)


  model = build_neural_network(config.embedding_dimension, config.fc_layer_size, config.num_layers_hidden, config.network_type, config.dropout)
  print(model)
  optimizer = build_optimizer(config.optimizer)

  auroc = EpochScoring(scoring=my_roc_auc, lower_is_better=False)

  net = NeuralNetBinaryClassifier(
      model,
      lr=config.learning_rate,
      max_epochs=config.max_epochs,
      batch_size=config.batch_size,
      criterion=torch.nn.modules.loss.BCEWithLogitsLoss,
      criterion__pos_weight=pos_weight,
      iterator_train=torch.utils.data.DataLoader,
      iterator_train__shuffle=True,
      optimizer=optimizer,
      optimizer__weight_decay=config.weight_decay,
      callbacks=[auroc, Checkpoint(monitor='my_roc_auc_best'), EarlyStopping(patience=5), WandbLogger(wandb_run)],
      train_split=lambda dataset: custom_split(dataset, train, val, sampling_strategy=config.class_imbalance_undersampling), 
      verbose=100,
      device=device,
  )

  net.fit(train_dataset, y=None)