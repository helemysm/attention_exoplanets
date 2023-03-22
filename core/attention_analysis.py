

import collections
import pickle

import matplotlib
import numpy as np
import seaborn as sns
import sklearn
import random
from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn import manifold
from box import box_from_file

import torch
from torch import nn


sns.set_style("whitegrid")


def data_iterator(data):
    '''
    Convert tensors to numpy
    '''
    for i, doc in enumerate(data):
        data_att = (doc[0]).squeeze(0).numpy()
        if i % 100 == 0 or i == len(data) - 1:
            print("{:.1f}% done".format(100.0 * (i + 1) / len(data)))
        yield [], data_att#
    
    
config = box_from_file(Path('config.yaml'), file_type='yaml')

data = np.load(config.dataset.attdata_path)    
    
# define arrays for entropies
cont=0
uniform_attn_entropy = 0
entropies = np.zeros((8, 8))
entropies_cls = np.zeros((8, 8)) 

'''
Compute the entropies for each head of each layer
'''
for tokens, attns in (data_iterator()):
    attns = 0.9999 * attns + (0.0001 / attns.shape[-1]) 
    
    uniform_attn_entropy -= np.log(1.0 / attns.shape[-1])
    
    # compute the eq 4.3.1
    entropies -= (attns * np.log(attns)).sum(-1).mean(-1)
    entropies_cls -= (attns * np.log(attns))[:, :, 0].sum(-1)
    cont = cont + 1

n_docs = len(data)
uniform_attn_entropy /= n_docs
entropies /= n_docs
entropies_cls /= n_docs


n_lay = 8

'''
This method obtains the average datapoints of each head for each layer of the network.
'''
def get_data_points(head_data):
    xs, ys, avgs = [], [], []
    for layer in range(n_lay):
        for head in range(n_lay):
            ys.append(head_data[layer, head])
            xs.append(1 + layer)
        avgs.append(head_data[layer].mean())
    return xs, ys, avgs

'''
This method computes the average of the attentions for a specific layer l_i and head k_head
'''

def avg_attention(atts_, h_k, l_i):
    '''
    params: attentions (model output)
    ----
    return:
    data_map_list_ = vector of avg attention for h_k
    '''
    data_map_signals = []

    for m in tqdm(range(len(atts_))):

        cont_true = cont_true+1
        data_map_list = []
        data_=atts_[m][0][l_i][h_k].squeeze(0).numpy()
        data_map = []
        for i in range(len(data_)):
            data_map.append(data_.T[i].mean())

        data_map_list.append(data_map)

        data_map_list = np.array(data_map_list).reshape(100,-1)

        data_map_list_ = []
        for i in range(len(data_map_list)):
            data_map_list_.append(data_map_list[i].mean())
    
    data_map_signals.append(data_map_list_)
    
    return data_map_signals    


#### 
xs, es, avg_es = get_data_points(entropies)
xs, es_cls, avg_es_cls = get_data_points(entropies_cls)

plt.figure(figsize=(12, 12))


'''
plot of avg entropies
'''
def plot_entropies(ax, data, avgs, label, c):
    ax.scatter(xs, data, c=c, s=25, label=label)
    ax.plot(1 + np.arange(8), avgs)
    ax.plot([1, 8], [uniform_attn_entropy, uniform_attn_entropy],
          c="k", linestyle="--")
    ax.text(3, uniform_attn_entropy - 0.3, "uniform attention",
          ha="center")
    ax.legend(loc="lower right")
    ax.set_ylabel("Avg. Attention Entropy") #(nats)
    ax.set_xlabel("Layer")



plot_entropies(plt.subplot(2, 1, 1), es, avg_es, "Heads", c=BLUE)

plt.show()
