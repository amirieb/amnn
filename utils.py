'''code for attentive multiveiw neural network (AMNN).

References:
    a. Hadi Amiri, Mitra Mohtarami, Isaac S. Kohane.
       "Attentive Multiview Text Representation for Differential Diagnosis"
       In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL'21).
'''

from sklearn.metrics import precision_score, recall_score, confusion_matrix
from functools import reduce
import re
import csv
import numpy as np
import json
import zipfile
import pickle


def read_data():
    query_train, document_train, labels_train = [None, None], [None, None], [None]
    query_val, document_val, labels_val = [None, None], [None, None], [None]
    # read training and validation data
    # store in query_train, document_train, labels_train, query_val, document_val, labels_val
    # ------------
    # (query_train[i],  document_train[i]) should contain representations of (query, document) pairs for the ith view
    # Example dimensions for the ith view:
    ## np.shape(query_train[i]): (?, 768)
    ## np.shape(document_train[i]): (?, 20000)
    # ------------
    # labels_. should contain class labels
    # Corresponding shape of labels_train:
    ## np.shape(labels_train) : (?, 2)

    return query_train, document_train, labels_train, query_val, document_val, labels_val


def get_precision_recall(y_probs, y_true):
    y_preds = [0 if val[0] > val[1] else 1 for val in y_probs]
    p_macro = precision_score(y_true, y_preds, average='macro')
    p_micro = precision_score(y_true, y_preds, average='micro')
    print(p_macro, p_micro)
    r_macro = recall_score(y_true, y_preds, average='macro')
    r_micro = recall_score(y_true, y_preds, average='micro')
    print(r_macro, r_micro)
    cf = confusion_matrix(y_true, y_preds)
    print(cf)
    

def flush_qrel(qrel):
    print('flushing qrel...')
    # [str(i), '0', mim, '1']
    f = open('qrel.txt', 'w')
    for l in qrel:
        f.write(' '.join(l) + '\n')
    f.close()
    
def flush_run(run):
    print('flushing run results...')
    # [i, 'Q0', mim, rank, prob, 'OMIM_run']
    f = open('run.txt', 'w')
    for l in run:
        f.write(' '.join([str(i) for i in l]) + '\n')
    f.close()
