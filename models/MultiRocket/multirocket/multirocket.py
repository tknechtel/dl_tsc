# Chang Wei Tan, Angus Dempster, Christoph Bergmeir, Geoffrey I Webb
#
# MultiRocket: Effective summary statistics for convolutional outputs in time series classification
# https://arxiv.org/abs/2102.00457

import time

import numpy as np
from sklearn.linear_model import RidgeClassifierCV

from ..multirocket import feature_names, get_feature_set
from models.MultiRocket.multirocket import minirocket as minirocket
#from multirocket import minirocket_multivariate as minirocket  # use multivariate version.
from models.MultiRocket.multirocket import rocket as rocket

from numba import njit, prange
from tqdm.auto import trange

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.metrics import jaccard_score





class Base_Classifier_MULTIROCKET(BaseEstimator, ClassifierMixin):
#multirocket default settings
    def __init__(self,
                 num_features=10_000,
                 feature_id=202,
                 kernel_selection=0,
                 verbose = True):
        """
        MultiRocket
        :param num_features: number of features
        :param feature_id: feature id to identify the feature combinations
        :param kernel_selection: 0 = minirocket kernels, 1 = rocket kernels
        """
        if kernel_selection == 0:
            self.name = "MiniRocket_{}_{}".format(feature_id, num_features)
        else:
            self.name = "Rocket_{}_{}".format(feature_id, num_features)

        self.kernels = None
        self.kernel_selection = kernel_selection
        self.num_features = num_features
        self.feature_id = feature_id
        self.verbose = verbose

        

        # get parameters based on feature id
        fixed_features, optional_features, num_random_features = get_feature_set(feature_id)
        self.fixed_features = fixed_features
        self.optional_features = optional_features
        self.num_random_features = num_random_features
        self.n_features_per_kernel = len(fixed_features) + num_random_features
        self.num_kernels = int(num_features / self.n_features_per_kernel)

        fixed_features_list = [feature_names[x] for x in self.fixed_features]
        optional_features_list = [feature_names[x] for x in self.optional_features]
        self.feature_list = fixed_features_list + optional_features_list
        print('FeatureID: {} -- features for each kernel: {}'.format(self.feature_id, self.feature_list))

        print('Creating {} with {} kernels'.format(self.name, self.num_kernels))
        self.classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10),
                                            normalize=True)

        self.train_duration = 0
        self.test_duration = 0
        self.generate_kernel_duration = 0
        self.apply_kernel_on_train_duration = 0
        self.apply_kernel_on_test_duration = 0

    def fit(self, x_train, y_train, predict_on_train=True):
        start_time = time.perf_counter()
        print(x_train.shape[1])
        print(x_train.shape)
        print(x_train.shape[-1])
        #print(x_train.shape[2])
        #print(x_train)
        print("x_train")

        print('[{}] Training with training set of {}'.format(self.name, x_train.shape))
        if self.kernel_selection == 0:
            # swap the axes for minirocket kernels. will standardise the axes in future.
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]))
            x_train = x_train.astype(np.float32)
            _start_time = time.perf_counter()
            self.kernels = minirocket.fit(x_train,
                                          self.feature_id,
                                          self.n_features_per_kernel,
                                          num_features=self.num_kernels)
            self.generate_kernel_duration = time.perf_counter() - _start_time

            x_train_transform = minirocket.transform(x_train, self.kernels)
            self.apply_kernel_on_train_duration = time.perf_counter() - _start_time - self.generate_kernel_duration
        else:
            #x_train = x_train.swapaxes(0, 1)
            _start_time = time.perf_counter()
            self.kernels = rocket.generate_kernels(x_train.shape[1],
                                                   num_kernels=self.num_kernels,
                                                   feature_id=self.feature_id,
                                                   num_features=self.n_features_per_kernel,
                                                   num_channels=x_train.shape[2])
            self.generate_kernel_duration = time.perf_counter() - _start_time

            x_train_transform = rocket.apply_kernels(x_train, self.kernels, self.feature_id)
            self.apply_kernel_on_train_duration = time.perf_counter() - start_time - self.generate_kernel_duration

        x_train_transform = np.nan_to_num(x_train_transform)

      

        elapsed_time = time.perf_counter() - start_time
        self.train_timings_ = [time.perf_counter() - start_time]
        print('Kernels applied!, took {}s'.format(elapsed_time))
        print('Transformed Shape {}'.format(x_train_transform.shape))

        print('Training')
        _start_time = time.perf_counter()
        self.classifier.fit(x_train_transform, y_train)
        self.train_duration = time.perf_counter() - _start_time
        #self.train_timings_ = [time.perf_counter() - start_time]

        self.train_timings_.append(time.perf_counter() - start_time)

        print('Training done!, took {:.3f}s'.format(self.train_duration))
        if predict_on_train:
            yhat = self.classifier.predict(x_train_transform)
        else:
            yhat = None

        return yhat

    def predict(self, x):
        #self.test_timings_ = self.test_timings_[0]
        print('[{}] Predicting'.format(self.name))
        start_time = time.perf_counter()

        if self.kernel_selection == 0:
            # swap the axes for minirocket kernels. will standardise the axes in future.
            #x = x.swapaxes(1, 2)
            x = x.reshape((x.shape[0], x.shape[1]))
            x = x.astype(np.float32)
            x_test_transform = minirocket.transform(x, self.kernels)
        else:
            #x = x.swapaxes(0, 1)
            x_test_transform = rocket.apply_kernels(x, self.kernels, self.feature_id)

        self.apply_kernel_on_test_duration = time.perf_counter() - start_time
        x_test_transform = np.nan_to_num(x_test_transform)

        print('Kernels applied!, took {:.3f}s. Transformed shape: {}. '.format(self.apply_kernel_on_test_duration,
                                                                               x_test_transform.shape))

        yhat = self.classifier.predict(x_test_transform)
        self.test_duration = time.perf_counter() - start_time
        self.test_timings_ = [time.perf_counter() - start_time]
        self.test_timings_.append(time.perf_counter() - start_time)

        print("[{}] Predicting completed, took {:.3f}s".format(self.name, self.test_duration))

        return yhat

    def score(self, x, y):
        
        print('[{}] Predicting'.format(self.name))
        start_time = time.perf_counter()

        if self.kernel_selection == 0:
            # swap the axes for minirocket kernels. will standardise the axes in future.
            #x = x.swapaxes(1, 2)
            x = x.reshape((x.shape[0], x.shape[1]))
            x = x.astype(np.float32)
            x_test_transform = minirocket.transform(x, self.kernels)
        else:
            x_test_transform = rocket.apply_kernels(x, self.kernels, self.feature_id)

        self.apply_kernel_on_test_duration = time.perf_counter() - start_time
        x_test_transform = np.nan_to_num(x_test_transform)

        print('Kernels applied!, took {:.3f}s. Transformed shape: {}. '.format(self.apply_kernel_on_test_duration,
                                                                               x_test_transform.shape))
        end_time = time.perf_counter()
        yhat = self.classifier.predict(x_test_transform)
        self.test_duration = time.perf_counter() - start_time
        self.test_timings_ = [time.perf_counter() - start_time]
        self.test_timings_.append(end_time - start_time)
        print("[{}] Predicting completed, took {:.3f}s".format(self.name, self.test_duration))

        y_true = y
        acc = accuracy_score(y_true, yhat)


        return acc


############################################

