# dl_tsc

***Deep Learning for time series classification - A review and experimental study***

## Results

### UCR Archive

* [Raw performance metrics](results/results.csv)

### [Requirements](requirements.txt)

scipy==1.4.1
numpy==1.18.5
tqdm==4.56.0
catch22== 0.1.0
numba==0.50.1
matplotlib==3.3.4
pandas==1.2.3
sktime==0.4.3
tensorflow-gpu==2.3.0
scikit_learn==0.24.1
tbb==2021.2.0

* Python;
* Matplotlib
* Numba;
* NumPy;
* Pandas;
* scikit-learn;
* sktime;
* scipy;
* TensorFlow-GPU;
* tqdm;
* catch22;
* tbb.

## Code
The code is divided as follows: 
* The [main.py](https://github.com/tknechtel/dl_tsc/blob/main/main.py) python file contains the necessary code to run an experiement. 
* The [utils](https://github.com/tknechtel/dl_tsc/blob/main/utils) folder contains the necessary functions to read the datasets and visualize the plots.
* The [classifiers](https://github.com/tknechtel/dl_tsc/tree/main/models) folder contains the different classifiers including: Rocket, MiniRocket, MultiRocket, InceptionTime, ResNet...

## Usage

### [`main.py`](main.py)

```
Arguments:
-d --dataset_names          : dataset names (optional, default=all)
-c --classifier_names       : classifier (optional, default=all)
-o --output_path            : path to results (optional, default=root_dir)
-i --iterations             : number of runs (optional, default=3)
-g --generate_results_csv   : make results.csv (optional, default=False)

Examples:
> python main.py
> python main.py -d Adiac Coffee -c multirocket_default -i 1
> python main.py -g True
```
The framework expects data from the UCR archive in the .ts format.

The folder structure for the datasets is as follows: <root>/UCRArchive_2018/dataset_name/
  
For example, the train/test of Adiac should be saved under /UCRArchive_2018/Adiac/


Calling main.py without any arguments trains every model on every dataset.


Results are saved in <root>/results.
  

To generate a results.csv for the tested models, main.py -g True is called.
  
### Critical difference diagrams
If you would like to generate such a diagram, take a look at [this code](https://github.com/hfawaz/cd-diagram)!
