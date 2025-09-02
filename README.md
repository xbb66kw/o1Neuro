# o1Neuro
A boosted neural network model for tabular data regression



## Installation

1. Install Anaconda Distribution following the instraction here: <https://docs.anaconda.com/free/anaconda/install/mac-os/>

2. Download the directory `o1Neuro` and put it in `YOURPATH`.

3. Run the following code in your terminal to set up your conda environment. The name of your new conda environment is `myenvironment`.

```conda create -n myenvironment python=3.10.12```

4. Run the following code to activate your conda environment:

```conda activate myenvironment```

5. Install the dependencies of o1Neuro under your conda environment using the following commands:

```conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision torchaudio pytorch-cuda=11.8 xgboost==1.5.0 numpy scikit-learn hyperopt scipy numba```

```pip install pytorch-tabnet```

6. Run the following code in your terminal:

```python3  /YOURPATH/o1Neuro/scripts/Example.py```

For more information about conda environment: <https://conda.io/projects/conda/en/latest/user-guide/getting-started.html#>

Alternatively, you can use IDEs like Spyder in conda, which provide friendly tools for visual learners.




## Reference

Chien-Ming Chi (2025) Constructive Universal Approximation and Sure Convergence for Multi-Layer Neural Networks.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact Information

Chien-Ming CHI

Institute of Statistical Science

Academia Sinica

xbbchi<span>@</span>stat.sinica.edu.tw


