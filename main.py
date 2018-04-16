import numpy as np
import math
from sklearn.manifold import TSNE

import matplotlib.colors as c
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from subprocess import call

from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization
from keras import optimizers
from keras.models import load_model

from keras.datasets import mnist

def main():
    # Parameters: whether to train a new neural network (or load already-trained from disk),
    # how many to train, how many to predict (or load from disk) and visualize (dimension? whether to build 3d .gif?)
    train_new = False
    n_train = 60000
    predict_new = False
    n_predict = 6000
    vis_dim = 3
    build_anim = False

    # Load MNIST dataset.
    x_train, y_train, x_test = import_format_data()

    # Build and fit autoencoder
    if train_new:
        autoencoder, encoder = build_autoencoder((x_train.shape[1],), encoding_dim=30)
        autoencoder.compile(optimizer=optimizers.Adadelta(), loss='mean_squared_error')

        autoencoder.fit(x_train[:n_train], x_train[:n_train], epochs=25, batch_size=32)
        autoencoder.save('%s_train_autoencoder1.h5' % (n_train))
        encoder.save('%s_train_encoder1.h5' % (n_train))
    else:
        encoder = load_model('60000_train_encoder1.h5')

    # Encode a number of MNIST digits, then perform t-SNE dim-reduction.
    if predict_new:
        x_train_predict = encoder.predict(x_train[:n_predict])

        print "Performing t-SNE dimensionality reduction..."
        x_train_encoded = TSNE(n_components=vis_dim).fit_transform(x_train_predict)
        np.save('%sx_%sdim_tnse_%s.npy' % (n_predict, vis_dim, n_train), x_train_encoded)
        print "Done."
    else:
        x_train_encoded = np.load(str(n_predict) + 'x_' + str(vis_dim) + 'dim_tnse_' + str(n_train) + '.npy')

    # Visualize result.
    vis_data(x_train_encoded, y_train, vis_dim, n_predict, n_train, build_anim)

def import_format_data():
    # Get dataset
    (x_train, y_train), (x_test, _) = mnist.load_data()

    # Turn [0,255] values in (N, A, B) array into [0,1] values in (N, A*B) flattened arrays
    x_train = x_train.astype('float64') / 255.0
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
    x_test = x_test.astype('float64') / 255.0
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]*x_test.shape[2]))

    return x_train, y_train, x_test

def build_autoencoder(input_shape, encoding_dim):
    # Activation function
    encoding_activation = 'tanh'
    decoding_activation = 'tanh'

    # Preliminary parameters
    inputs = Input(shape=input_shape)
    feat_dim = input_shape[0]

    # kernel_initializer='lecun_normal'
    # Encoding layers: successive smaller layers, then a batch normalization layer.
    encoding = Dense(feat_dim, activation=encoding_activation)(inputs)
    encoding = BatchNormalization()(encoding)
    encoding = Dense(feat_dim/2, activation=encoding_activation)(encoding)
    encoding = BatchNormalization()(encoding)
    encoding = Dense(feat_dim/4, activation=encoding_activation)(encoding)
    encoding = BatchNormalization()(encoding)
    encoding = Dense(encoding_dim, activation=encoding_activation)(encoding)
    encoding = BatchNormalization()(encoding)

    # Decoding layers for reconstruction
    decoding = Dense(feat_dim/4, activation=decoding_activation)(encoding)
    decoding = BatchNormalization()(decoding)
    decoding = Dense(feat_dim/2, activation=decoding_activation)(decoding)
    decoding = BatchNormalization()(decoding)
    decoding = Dense(feat_dim, activation=decoding_activation)(decoding)
    
    # Return the whole model and the encoding section as objects
    autoencoder = Model(inputs, decoding)
    encoder = Model(inputs, encoding)

    return autoencoder, encoder

def vis_data(x_train_encoded, y_train, vis_dim, n_predict, n_train, build_anim):
    cmap = plt.get_cmap('rainbow', 10)

    # 3-dim vis: show one view, then compile animated .gif of many angled views
    if vis_dim == 3:
        # Simple static figure
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        p = ax.scatter3D(x_train_encoded[:,0], x_train_encoded[:,1], x_train_encoded[:,2], 
                c=y_train[:n_predict], cmap=cmap, edgecolor='black')
        fig.colorbar(p, drawedges=True)
        plt.show()

        # Build animation from many static figures
        if build_anim:
            angles = np.linspace(180, 360, 20)
            i = 0
            for angle in angles:
                fig = plt.figure()
                ax = plt.axes(projection='3d')
                ax.view_init(10, angle)
                p = ax.scatter3D(x_train_encoded[:,0], x_train_encoded[:,1], x_train_encoded[:,2], 
                        c=y_train[:n_predict], cmap=cmap, edgecolor='black')
                fig.colorbar(p, drawedges=True)
                outfile = 'anim/3dplot_step_' + chr(i + 97) + '.png'
                plt.savefig(outfile, dpi=96)
                i += 1
            call(['convert', '-delay', '50', 'anim/3dplot*', 'anim/3dplot_anim_' + str(n_train) + '.gif'])

    # 2-dim vis: plot and colorbar.
    elif vis_dim == 2:
        plt.scatter(x_train_encoded[:,0], x_train_encoded[:,1], 
                c=y_train[:n_predict], edgecolor='black', cmap=cmap)
        plt.colorbar(drawedges=True)
        plt.show()

if __name__ == '__main__':
    main()