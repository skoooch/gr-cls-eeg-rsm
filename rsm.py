import numpy as np
import torch
import torch.nn as nn
import matplotlib.pylab as plt
from tqdm import tqdm
from process_data import get_data
import sys
import os
from PIL import Image
from scipy.spatial.distance import squareform, pdist
from sklearn import manifold, datasets
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
class MDS:
    """ Classical multidimensional scaling (MDS)
                                                                                               
    Args:                                                                               
        D (np.ndarray): Symmetric distance matrix (n, n).          
        p (int): Number of desired dimensions (1<p<=n).
                                                                                               
    Returns:                                                                                 
        Y (np.ndarray): Configuration matrix (n, p). Each column represents a 
            dimension. Only the p dimensions corresponding to positive 
            eigenvalues of B are returned. Note that each dimension is 
            only determined up to an overall sign, corresponding to a 
            reflection.
        e (np.ndarray): Eigenvalues of B (p, ).                                                                     
                                                                                               
    """    
    def cmdscale(D, p = None):
        # Number of points                                                                        
        n = len(D)
        # Centering matrix                                                                        
        H = np.eye(n) - np.ones((n, n))/n
        # YY^T                                                                                    
        B = -H.dot(D**2).dot(H)/2
        # Diagonalize                                                                             
        evals, evecs = np.linalg.eigh(B)
        # Sort by eigenvalue in descending order                                                  
        idx   = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:,idx]
        # Compute the coordinates using positive-eigenvalued components only                      
        w, = np.where(evals > 0)
        L  = np.diag(np.sqrt(evals[w]))
        V  = evecs[:,w]
        Y  = V.dot(L)   
        if p and Y.shape[1] >= p:
            return Y[:, :p], evals[:p]
        return Y, evals
    def two_mds(D,p=None):
        my_scaler = manifold.MDS(n_jobs=-1, n_components=2)
        return my_scaler.fit_transform(D)
    def three_mds(D,p=None):
        my_scaler = manifold.MDS(n_jobs=-1, n_components=3)
        return my_scaler.fit_transform(D)
times = [(25.390625,99.609375),(99.609375, 175.78125), (175.78125, 250)]
task = 'cls'
for i in range(3):
    data = get_data(task)
    labels = data.keys()
    activations_flat = []
    timepoints = np.loadtxt('data/timepoints_8_%s.csv' % task, delimiter=',') 
    
    time_period = (np.where(timepoints == times[i][0])[0][0], np.where(timepoints == times[i][1])[0][0])
    points_per_object = {}
    for cat in labels:
        points_per_object[cat] = 0
        for object_data in data[cat]:
            for eeg in object_data:
                relevant_signal = eeg[time_period[0]:time_period[1], :]
                activations_flat.append(relevant_signal.flatten())
                points_per_object[cat] += 1
    act_array = np.asarray(activations_flat)
    result = squareform(pdist(act_array, metric="correlation"))
    # embedding = MDS.cmdscale(result, 2)[0]
    # embedding = {cat:embedding[i*num_images_per_label:(i+1)*num_images_per_label] # split into categories
    #             for i, cat in enumerate(labels)}   
    # ax = plt.gca()
    # ax.set_xticks([])
    # ax.set_yticks([])
    # for cat in labels:
    #     ax.scatter(embedding[cat][:, 0],
    #                 embedding[cat][:, 1],
    #                 label = cat)
    # ax.legend()
    # plt.savefig('vis/rsm/rgb_1.png')    
    # plt.clf()
    # ax = plt.gca()
    # ax.set_xticks([])
    # ax.set_yticks([])
    # for cat in labels:
    #     avr_x = np.mean(embedding[cat][:, 0])
    #     avr_y = np.mean(embedding[cat][:, 1])
    #     ax.scatter(avr_x,
    #                 avr_y,
    #                 label = cat)
    # ax.legend()
    # plt.savefig('vis/rsm/rgb_1_avr.png')  
    embedding = MDS.two_mds(result) 
    total_objects_sofar = 0
    embedding_categorized = {}
    for cat in labels:
        embedding_categorized[cat] = embedding[total_objects_sofar:total_objects_sofar + points_per_object[cat]]
        total_objects_sofar += points_per_object[cat]
    fig = plt.figure()
    ax = fig.add_subplot()
    for cat in labels:
        ax.scatter(embedding_categorized[cat][:, 0],
                    embedding_categorized[cat][:, 1],
                    label=cat)
        
    # for cat in labels:
    #     for i in range(len(embedding[cat][:, 0])):
    #         ax.text(embedding[cat][i, 0],
    #                     embedding[cat][i, 1],
    #                     i)
    ax.legend()
    plt.title("%sms to %sms grasp (correlation)" % (times[i][0], times[i][1]))
    plt.savefig('vis/%s/ts%s_correlation.png' % (task,(i+1)))   
    plt.clf()

    fig = plt.figure()
    ax = fig.add_subplot()
    for cat in labels:
        avr_x = np.mean(embedding_categorized[cat][:, 0])
        avr_y = np.mean(embedding_categorized[cat][:, 1])
        ax.scatter(avr_x,
                    avr_y,
                    label=[cat])
    ax.legend()
    plt.title("%sms to %sms Averaged grasp (correlation)" % (times[i][0], times[i][1]))
    plt.savefig('vis/%s/ts%s_correlation_avr.png' % (task, (i+1)))   
    plt.clf()