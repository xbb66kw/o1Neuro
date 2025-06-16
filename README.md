# o1Neuro
Prediction model for tabular data



## Installation

1. Install Anaconda Distribution following the instraction here: <https://docs.anaconda.com/free/anaconda/install/mac-os/>

2. Download the directory `o1Neuro` and put it in `YOURPATH`.

3. Run the following code in your terminal to set up your conda environment. The name of your new conda environment is `myenvironment`.

```conda create -n myenvironment python=3.10.12```

4. Run the following code to activate your conda environment:

```conda activate myenvironment```

5. Install the dependencies of Collaborative Trees Ensemble under your conda environment using the following command:

```conda install xgboost==1.5.0 numpy conda-forge::scikit-learn conda-forge::hyperopt scipy```

6. Run the following code in your terminal:

```python3 YOURPATH/o1Neuro/o1Neuro/example.py```

For more information about conda environment: <https://conda.io/projects/conda/en/latest/user-guide/getting-started.html#>

Alternatively, you can use IDEs like Spyder in conda, which provide friendly tools for visual learners.

## Dependencies

The following are my environments:

* Spyder version: 5.5.1 (conda)
* Python version: 3.10.12 64-bit
  
  _Early versions of Python do not support type hints. The package `hyperopt` may not be compatible with the latest Python version. :c_
  
* Conda version: 24.3.0

* xgboost: Used for boosting algorithms (version 1.5.0).
* sklearn: Provides machine learning algorithms and utilities (version 1.2.2).
* scipy: Scientific computing library for numerical operations (version 1.11.1).
* hyperopt: Library for hyperparameter optimization (version 0.2.7).




## Reference

Chien-Ming Chi (2025) Sure Convergence and Constructive Universal Approximation for Multi-Layer Neural Networks.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact Information

Chien-Ming CHI

Institute of Statistical Science

Academia Sinica

xbbchi<span>@</span>stat.sinica.edu.tw


