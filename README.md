Until graph-tool version 2.92 is available via conda, the installation is complicated.
# Installing graph-tool in a Singularity Container

## Step 1: Create a Directory for Installation
Create a directory to store all installation-related files:
```bash
mkdir singularity
cd singularity
```

## Step 2: Pull the Development Version of graph-tool
Use Singularity to pull the latest development version of graph-tool:
```bash
singularity pull docker://tiagopeixoto/graph-tool-git
```

## Step 3: Bind Required Data and Access the Container
Bind the necessary directories and access the container shell:
```bash
singularity shell -B /data/bionets/datasets/scrnaseq_ji:/mnt/data \
                 -B /data/bionets/je30bery:/mnt/je30bery \
                 graph-tool-git_latest.sif
```

## Step 4: Create a Virtual Environment
Since `pip` is not installed in the container, use Python to create a virtual environment:
```bash
python3 -m venv /data/bionets/je30bery/singularity/gt
```
Activate the virtual environment:
```bash
source /data/bionets/je30bery/singularity/gt/bin/activate
```

## Step 5: Create a Symbolic Link to the graph-tool Installation
Navigate to the site-packages directory of your virtual environment:
```bash
cd /data/bionets/je30bery/singularity/gt/lib/python3.13/site-packages
```
Create a symbolic link to the graph-tool installation inside the container:
```bash
ln -s /usr/lib/python3.13/site-packages/graph_tool/ graph_tool
```

## Step 6: Verify Installation
Check that graph-tool is correctly installed by running:
```bash
python -c "import graph_tool.all as gt; print(gt.__version__)"
```
If the installation was successful, this should print the installed version of graph-tool.

## Optional: Install Additional Dependencies
If you need additional Python packages, install them inside the virtual environment:
```bash
pip install numpy scipy matplotlib
```

Now, graph-tool should be ready to use within the Singularity container.









Originally, the installation worked as follows:

Python Implementation of https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Multivariate-distributions.pdf

Clone this repository. Create a conda environment with python 3.9. 

```
conda create --name rb python=3.9
conda activate rb
```

Within the environment, install the required packages using pip and conda:
```
pip install -r requirements.txt
```
```
conda install -c conda-forge graph-tool
```
If you want to perform the distance matrix calculation on a GPU, also install [pylibraft](https://anaconda.org/rapidsai/pylibraft) and [cupy](https://docs.cupy.dev/en/v13.2.0/install.html#installing-cupy). 
This step is not helpful when using mahalanobis distance, because it will fallback to cpu distance matrix calulcation anyways.
```
 conda install rapidsai::pylibraft
 conda install -c conda-forge cupy
```

