import timeit
from node2vec import Node2Vec
import numpy as np


def embedding_func(dimensions,walk_length, G_embeddings, EMBEDDING_FILENAME):
  # Generate walks
  node2vec = Node2Vec(G_embeddings, dimensions=dimensions, walk_length=walk_length, num_walks=50, workers=1)

  # train node2vec model
  n2w_model = node2vec.fit(window=7, min_count=1)

  # Save embeddings for later use
  n2w_model.wv.save_word2vec_format(EMBEDDING_FILENAME+'_embeddings_dim:'+str(dimensions)+'_len:'+str(walk_length))

  # Save model for later use
  n2w_model.save(EMBEDDING_FILENAME+'_embedding_model_dim:'+str(dimensions)+'_len:'+str(walk_length))

  return


def grid_search(parameters, func):
    def combination(params):
        if len(params) > 1:
            return [prev + [(params[-1][0], el)] for el in params[-1][1] for prev in combination(params[:-1])]
        else:
            return [[(params[0][0], el)] for el in params[0][1]]
    
    parameters= [dict(comb) for comb in combination(list(parameters.items()))]

    times = {}
    scores = {}
    for params in parameters:

        #start timer
        start = timeit.default_timer()

        #store the scores for these parameters
        scores[tuple(params.items())] = func(**params)

        #end timer
        stop = timeit.default_timer()

        #store time required
        times[tuple(params.items())] = stop - start
        print('Time for '+str(params)+' is: ' + str(stop - start))

    return times, scores


def retrieve_embeddings(DIR):
    with open(DIR) as f:
        lines = f.readlines()
    lines2 = [line.split(' ') for line in lines[1:]]
    embeddings = dict([(line[0], np.asarray([float(el) for el in line[1:]]))for line in lines2])
    return embeddings