# GRAINet: Mapping grain size distributions in river beds from UAV images with convolutional neural networks

This is a demonstration to show how to run the training and testing code used to analyze the grain size distributions in river beds over entire gravel bars ([Lang et al., 2020]( https://doi.org/10.5194/hess-2020-196)).

Along with the code, we release a small subset of the dataset used in the study, which is a single gravel bar (orthophoto) with 212 manually annotated image tiles. The full dataset cannot be published for commercial reasons, as it is owned by a private company (who also created it at their own cost).

**Important note**: The trained model resulting from this demo will not generalize as described in the paper.


## Getting Started

1) Clone this repository to your local machine. In your terminal type:
    ```
    git clone URL
    ```
   
2) Download the data from [here](https://share.phys.ethz.ch/~pf/nlangdata/GRAINet_demo_data.zip).

    Move the data folder into the GRAINet directory. The directory tree should look like this:
    GRAINet/data/


## Prerequisites
This code uses keras with a tensorflow backend. GDAL is used to predict for georeferenced orthophotos.
The following instructions will guide you to install:

* python3
* jupyter
* tensorflow
* keras
* gdal

## Installing
We recommend to install python via anaconda and to create a new conda environment.

1) [Install Anaconda](https://docs.anaconda.com/anaconda/install/) and read the [Anaconda tutorial](https://conda.io/docs/user-guide/getting-started.html)

2) Create a new environment: ```conda create --name GRAINenv python=3.7.1```

3) Activate the new conda environment (for conda 4.6 and later versions)
    * Windows: ```conda activate GRAINenv```
    * Linux and macOS: ```conda activate GRAINenv```
    
    For versions prior to conda 4.6, use:
    * Windows: ```activate GRAINenv```
    * Linux, macOS: ```source activate GRAINenv```
    
    --> now your terminal prompt should start with ***(GRAINenv)*** 
    
4) Install the following packages in your activated GRAINenv:
    ```
    conda install jupyter
    conda install matplotlib
    conda install keras=2.2.4
    conda install h5py=2.9.0
    ```
    
5) Install ***tensorflow*** with anaconda or follow the [official tensorflow installation instructions (e.g. with pip)](https://www.tensorflow.org/install/pip)
    or check the [anaconda installation instructions](https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/).
    ```
    conda install tensorflow-gpu=1.13.1
    ```
      
6) Install ***GDAL*** with anaconda or follow the [official GDAL installation instructions](https://gdal.org/download.html)
   ```
   conda install -c conda-forge gdal
   ``` 

## Verify your installation

Run `python` in your conda GRAINenv. Then:
```
import keras
import tensorflow
import h5py
from osgeo import gdal
```

The code has been tested with the following versions:
```
keras.__version__
'2.2.4'
tensorflow.__version__
'1.13.1'
h5py.__version__
'2.9.0'
```

## Run the notebook
Open the jupyter notebook:
```
jupyter notebook GRAINet_demo_dm_regression.ipynb
```

## Citation

If you use this code please cite our paper: 

*Lang, Nico, Andrea Irniger, Agnieszka Rozniak, Roni Hunziker, Jan Dirk Wegner, and Konrad Schindler. "GRAINet: Mapping grain size distributions in river beds from UAV images with convolutional neural networks." Hydrology and Earth System Sciences Discussions (2020): 1-38.*

BibTex:

```
@article{lang2020grainet,
  title={GRAINet: Mapping grain size distributions in river beds from UAV images with convolutional neural networks},
  author={Lang, Nico and Irniger, Andrea and Rozniak, Agnieszka and Hunziker, Roni and Wegner, Jan Dirk and Schindler, Konrad},
  journal={Hydrology and Earth System Sciences Discussions},
  pages={1--38},
  year={2020},
  publisher={Copernicus GmbH}
}
```





