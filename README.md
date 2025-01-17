Python Implementation of https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Multivariate-distributions.pdf

Clone this repository. Create a conda environment with python 3.9. 

```
conda create --name rb python=3.9
conda activate rb
```

Within the environment, install the required packages using pip:
```
pip install -r requirements.txt
```

If you want to perform the distance matrix calculation on a GPU, also install [cupy](https://docs.cupy.dev/en/v13.2.0/install.html#installing-cupy). 
This step is not helpful when using mahalanobis distance, because it will fallback to cpu distance matrix calulcation anyways.
```
 conda install -c conda-forge cupy
```

