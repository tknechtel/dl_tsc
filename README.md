# dl_tsc

***Deep Learning for time series classification - A review and experimental study***

## Results

### UCR Archive

* [Raw performance metrics](results/results.csv)

### [Requirements](requirements.txt)

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
* Link to [Colab Notebook](Colab%20notebooks/dl_tsc_notebook.ipynb)

The code is divided as follows: 
* The [main.py](https://github.com/tknechtel/dl_tsc/blob/main/main.py) python file contains the necessary code to run an experiement. 
* The [utils](https://github.com/tknechtel/dl_tsc/blob/main/utils) folder contains the necessary functions to read the datasets and visualize the plots.
* The [classifiers](https://github.com/tknechtel/dl_tsc/tree/main/models) folder contains the different classifiers including: Rocket, MiniRocket, MultiRocket, InceptionTime, ResNet...

## Usage

### [`main.py`](main.py) in [Colab](Colab%20notebooks/dl_tsc_notebook.ipynb)

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
  
## References

* [UCR/UEA archive](http://timeseriesclassification.com/TSC.zip)
* [Rocket repository](https://github.com/angus924/rocket)
* [MiniRocket repository](https://github.com/angus924/minirocket)
* [MultiRocket repository](https://github.com/ChangWeiTan/MultiRocket)
* [InceptionTime repository](https://github.com/hfawaz/InceptionTime/)
* [dl_time_series_class repository](https://github.com/JakobSpahn/dl_time_series_class)
* [dl-4-tsc repository](https://github.com/hfawaz/dl-4-tsc)
  
 


