import numpy as np

X_fname = '../data/X_train6.npy'
y_fname = '../data/y_train6.npy'
X_train = np.load(X_fname)
y_train = np.load(y_fname)
#
# convnets = [[(32,3),(64,3),(128,3)]]
# dropouts = [0.4, 0.3, 0.2]
# densenets = [(256,2), (512,2), (1024,2)]
#
from cnn6 import cnn_architecture
from keras import backend as K

K.set_image_dim_ordering('th')

dropout = 0.1
convnets = [(32,3),(64,3),(128,3)]
dense = [512,2]
print('Loading data...')
cnn_architecture(X_train, y_train, conv_arch=convnets, dense=dense, dropout=dropout, batch_size=128, nb_epoch=20, dirpath = '../data/results/',patience=20)
#
# from maxout_cnn6 import cnn_architecture
#
# dropouts = [0.2]
# maxdensenets = [(256,2),(256,4),(512,2),(512,4)]
#
# for densenet in maxdensenets:
#     for dropout in dropouts:
#
#         convnets = [(32,3),(64,3),(128,3)]
#         print 'Loading data...'
#         cnn_architecture(X_train, y_train, conv_arch=convnets, maxdense=densenet, dropout=dropout, batch_size=256, nb_epoch=100, dirpath = '../data/results/')
