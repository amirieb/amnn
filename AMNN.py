'''code for attentive multiveiw neural network (AMNN).

References:
    a. Hadi Amiri, Mitra Mohtarami, Isaac S. Kohane.
       "Attentive Multiview Text Representation for Differential Diagnosis"
       In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL'21).

To do list:
    a. template to load data in utils.read_data() function
'''

import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.metrics import Precision, Recall
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from acl21_amnn.fuse import get_conv_fuse, get_cross_fuse, get_dot_fuse
from acl21_amnn.utils import read_data
import sys

BATCH_SIZE = 32
EPOCHS = 20
JOINT_SPACE_SIZE = 100
FILTER_SIZE = 250
KERNEL_SIZE = 3
LEARNING_RATE = 1e-5
VIEW_COUNT = 2  # >= 2
FUSE_TYPE = 'dot'  # possible values {'none', 'dot', 'conv', 'cross'}
if FUSE_TYPE not in {'none', 'dot', 'conv', 'cross'}:
    print(FUSE_TYPE, 'error!')
    sys.exit(1)

# read training and validation data
query_train, document_train, labels_train, query_val, document_val, labels_val = read_data()


'''build model'''
query_vec_length = [len(query_train[i][0]) for i in range(VIEW_COUNT)]
document_vec_length = [len(document_train[i][0]) for i in range(VIEW_COUNT)]
query_views = [None for _ in range(VIEW_COUNT)]
document_views = [None for _ in range(VIEW_COUNT)]
for i in range(VIEW_COUNT):
    query_views[i] = layers.Input(shape=(query_vec_length[i],), dtype='float32')
    document_views[i] = layers.Input(shape=(document_vec_length[i],), dtype='float32')


'''create a shared space'''
query_views_shared = [None for _ in range(VIEW_COUNT)]
document_views_shared = [None for _ in range(VIEW_COUNT)]
for i in range(VIEW_COUNT):
    query_views_shared[i] = layers.Dense(JOINT_SPACE_SIZE, activation='relu')(query_views[i])
    document_views_shared[i] = layers.Dense(JOINT_SPACE_SIZE, activation='relu')(query_views[i])


'''compute layer-wise attentions'''
query_views_attentive = [None for _ in range(VIEW_COUNT)]
document_views_attentive = [None for _ in range(VIEW_COUNT)]
attention_layers = [None for _ in range(VIEW_COUNT)]
attention_weights = [0. for _ in range(VIEW_COUNT)]
for i in range(VIEW_COUNT):
    attention_layers[i] = layers.dot([query_views_shared[i], document_views_shared[i]], axes=1)
sim_merged = layers.concatenate([attention_layers[i] for i in range(VIEW_COUNT)])
att = layers.Dense(VIEW_COUNT, activation='softmax', name='attention_layer')(sim_merged)
for i in range(VIEW_COUNT):
    # TESTED FOR TWO VIEWS (text and code views in the paper)..
    attention_weights[i] = tf.slice(att, [0, i], [-1, 1])
    query_views_attentive[i] = layers.multiply([attention_weights[i], query_views_shared[i]])
    document_views_attentive[i] = layers.multiply([attention_weights[i], document_views_shared[i]])


'''fuse & concatenate'''
if FUSE_TYPE == 'none':
    merged = layers.concatenate(
        [layers.concatenate([query_views_attentive[i], document_views_attentive[i]]) for i in range(VIEW_COUNT)])

if FUSE_TYPE == 'conv':  # conv
    views_conv = [None for _ in range(VIEW_COUNT)]
    for i in range(VIEW_COUNT):
        views_conv[i] = get_conv_fuse(FILTER_SIZE, KERNEL_SIZE, JOINT_SPACE_SIZE, query_views_attentive[i],
                                      document_views_attentive[i])
    merged = layers.concatenate(
        [layers.concatenate([query_views_attentive[i], document_views_attentive[i], views_conv[i]]) for i in
         range(VIEW_COUNT)])

if FUSE_TYPE == 'cross':  # cross
    views_cross = [None for _ in range(VIEW_COUNT)]
    for i in range(VIEW_COUNT):
        views_cross[i] = get_cross_fuse(JOINT_SPACE_SIZE, query_views_attentive[i], document_views_attentive[i])
    merged = layers.concatenate(
        [layers.concatenate([query_views_attentive[i], document_views_attentive[i], views_cross[i]]) for i in
         range(VIEW_COUNT)])

if FUSE_TYPE == 'dot':  # sim
    views_dot = [None for _ in range(VIEW_COUNT)]
    for i in range(VIEW_COUNT):
        views_dot[i] = get_dot_fuse(query_views_attentive[i], document_views_attentive[i])
    merged = layers.concatenate(
        [layers.concatenate([query_views_attentive[i], document_views_attentive[i], views_dot[i]]) for i in
         range(VIEW_COUNT)])


'''train model'''
my_callbacks = [
    EarlyStopping(monitor='val_loss', patience=2)]  # ,ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),]
preds = layers.Dense(2, activation='softmax')(merged)
model = Model([query_views[i] for i in range(VIEW_COUNT)] + [document_views[i] for i in range(VIEW_COUNT)], preds)
model.compile(optimizer=Adam(lr=LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=[Precision(), Recall()])
print(model.summary())
model.fit([query_train[i] for i in range(VIEW_COUNT)] + [document_train[i] for i in range(VIEW_COUNT)], labels_train,
          batch_size=BATCH_SIZE, validation_data=(
    [query_val[i] for i in range(VIEW_COUNT)] + [document_val[i] for i in range(VIEW_COUNT)], labels_val),
          epochs=EPOCHS, callbacks=my_callbacks)