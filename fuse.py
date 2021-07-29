'''code for attentive multiveiw neural network (AMNN).

References:
    a. Hadi Amiri, Mitra Mohtarami, Isaac S. Kohane.
       "Attentive Multiview Text Representation for Differential Diagnosis"
       In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL'21).
'''

from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, AveragePooling1D

def get_conv_fuse(FILTER_SIZE, KERNEL_SIZE, vector_len, t1, t2):
    intm1 = layers.RepeatVector(vector_len)(t1)
    intm2 = layers.RepeatVector(vector_len)(t2)
    intm3 = layers.Permute((2, 1))(intm2)
    intm4 = layers.multiply([intm1, intm3])
    intm5 = Conv1D(FILTER_SIZE, KERNEL_SIZE, padding='valid', activation='relu', strides=1)(intm4)
    intm6 = AveragePooling1D()(intm5)
    conv_fuse = layers.Flatten()(intm6)
    return conv_fuse


def get_cross_fuse(JOINT_SPACE_SIZE, t1, t2):
    query_jspace = layers.Dense(JOINT_SPACE_SIZE, activation='relu')(t1)
    doc_jspace = layers.Dense(JOINT_SPACE_SIZE, activation='relu')(t2)
    intm1 = layers.RepeatVector(JOINT_SPACE_SIZE)(query_jspace)
    intm2 = layers.RepeatVector(JOINT_SPACE_SIZE)(doc_jspace)
    intm3 = layers.Permute((2, 1))(intm2)
    intm4 = layers.multiply([intm1, intm3])
    cross_fuse = layers.Flatten()(intm4)
    return cross_fuse


def get_dot_fuse(t1, t2):
    dot_fuse = layers.dot([t1, t2], axes=1)
    return dot_fuse
