import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time

from utils.utils import save_logs
from utils.utils import calculate_metrics

import matplotlib 
#matplotlib.use('agg') #To use non-gui backend
import matplotlib.pyplot as plt 

from tqdm.keras import TqdmCallback

class Classifier_MLP:

    def __init__(self, output_dir, input_shape, nb_classes, verbose=False, build=True):
        self.output_dir = output_dir
        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if(verbose == True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_dir + 'model_init.hdf5')
        return

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        # flatten/reshape because when multivariate all should be on the same axis
        input_layer_flattened = keras.layers.Flatten()(input_layer)

        layer_1 = keras.layers.Dropout(0.1)(input_layer_flattened)
        layer_1 = keras.layers.Dense(500, activation='relu')(layer_1)

        layer_2 = keras.layers.Dropout(0.2)(layer_1)
        layer_2 = keras.layers.Dense(500, activation='relu')(layer_2)

        layer_3 = keras.layers.Dropout(0.2)(layer_2)
        layer_3 = keras.layers.Dense(500, activation='relu')(layer_3)

        output_layer = keras.layers.Dropout(0.3)(layer_3)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=0.0001)

        file_path = self.output_dir + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
            save_best_only=True)

        self.callbacks = [reduce_lr,model_checkpoint,TqdmCallback(verbose=0)] 

        return model

    def fit(self, x_train, y_train, x_val, y_val,y_true):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training
        batch_size = 16
        nb_epochs = 5000

        mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
            verbose=False, validation_data=(x_val,y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_dir + 'last_model.hdf5')

        model = keras.models.load_model(self.output_dir+'best_model.hdf5')

        y_pred = model.predict(x_val)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred , axis=1)

        save_logs(self.output_dir, hist, y_pred, y_true, duration, self.verbose)

        keras.backend.clear_session()