Python Implementation of https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Multivariate-distributions.pdf


Create a conda environment with python 3.9. Within the environment, install the following:
```
anndata==0.10.9
networkx==3.2.1
numpy==1.26.4
pandas==2.2.3
scipy==1.13.1
```

If you want to perform the distance matrix calculation on a GPU, also install [cupy](https://docs.cupy.dev/en/v13.2.0/install.html#installing-cupy).
```
 conda install -c conda-forge cupy
```

