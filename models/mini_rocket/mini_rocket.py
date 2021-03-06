import time
import numpy as np
import pandas as pd
#from .base.base_rocket import 
from utils.utils import calculate_metrics
from .base.base_mini_rocket import Base_Classifier_MINIROCKET
from sklearn.linear_model import RidgeClassifierCV

class Classifier_MINIROCKET():
    def __init__(self, output_dir, verbose):
        self.output_dir = output_dir
        self.verbose = verbose

        # Hyperparameters
        #self.n_kernels = 10_000
        #self.kss = (7, 9, 11)

    def fit(self, x_train, y_train, x_test, y_test, y_true):
        # Reshape x_train and x_test so it fits the classifier
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]))

        # Get non-oneHot-encoded labels
        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)

        

        ###########################
        ##Train ROCKET Classifier##
        ###########################
        start_time = time.time()
        #minirocket = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
        minirocket = Base_Classifier_MINIROCKET(self.verbose)
        minirocket.fit(x_train, y_train)
        duration = time.time() - start_time
        train_timings = minirocket.train_timings_

        ##########################
        ##Test ROCKET Classifier##
        ##########################
        
        acc = minirocket.score(x_test, y_test)
        test_timings = minirocket.test_timings_

        #############################
        ##Predict ROCKET Classifier##
        #############################
        y_pred = minirocket.predict(x_test)

        # Save Metrics
        df_metrics = calculate_metrics(y_test, y_pred, duration)
        if self.verbose:
            print(df_metrics)
        df_metrics.to_csv(self.output_dir +
                          'df_metrics.csv', index=False)

        # Save train and test timings
        train_timings.append(train_timings[0]+train_timings[1])
        test_timings.append(test_timings[0]+test_timings[1])
        df_timings = pd.DataFrame([train_timings, test_timings], index=[
                                  'train', 'test'], columns=['x_transform', 'ridge_operation', 'total'])
        if self.verbose:
            print(df_timings)
        df_timings.to_csv(self.output_dir + 'df_timings.csv', index=False)
