import time
import numpy as np
import pandas as pd
from .multirocket.multirocket_best_v1 import Base_Classifier_MULTIROCKET
from utils.utils import calculate_metrics


class Classifier_MULTIROCKET():
    def __init__(self, output_dir, verbose):
        self.output_dir = output_dir
        self.verbose = verbose

        # Hyperparameters
        self.num_features=20000,
        self.feature_id=50,
        self.kernel_selection=0

    def fit(self, x_train, y_train, x_test, y_test, y_true):
        # Reshape x_train and x_test so it fits the classifier
        #x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]))
        #x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]))

        # Get non-oneHot-encoded labels
        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)

        ###########################
        ##Train ROCKET Classifier##
        ###########################
        start_time = time.time()
        multirocket = Base_Classifier_MULTIROCKET()
        multirocket.fit(x_train, y_train)
        duration = time.time() - start_time
        train_timings = multirocket.train_timings_

        ##########################
        ##Test ROCKET Classifier##
        ##########################
        acc = multirocket.score(x_test, y_test)
        test_timings = multirocket.test_timings_

        #############################
        ##Predict ROCKET Classifier##
        #############################
        y_pred = multirocket.predict(x_test)

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
