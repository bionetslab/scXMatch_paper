Python Implementation of https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Multivariate-distributions.pdf
Clone this repository. Create a conda environment with python 3.10. 
```
conda create --name scxmatch python=3.10
conda activate scxmatch
```

Within the environment, install the required packages using pip and conda:
```
conda install -c conda-forge "graph-tool==2.9.2"
```
```
pip install -r requirements.txt
```
If you want to perform the distance matrix calculation on a GPU, also install [pylibraft](https://anaconda.org/rapidsai/pylibraft) and [cupy](https://docs.cupy.dev/en/v13.2.0/install.html#installing-cupy). 
This step is not helpful when using mahalanobis distance, because it will fallback to cpu distance matrix calulcation anyways.
```
 conda install rapidsai::pylibraft
 conda install -c conda-forge cupy
```

