# Score Based Constrained Generation
Research project on costrained generation with unconditional score based models

## Installation
We suggest creating a python virtual environment with Python3.10, then after activating the environment type:
```
make install
```
to install all the requirements.

## Datasets availability
All datasets are publicly available.
Datasets are generated (eventually downloaded) on-the-fly and then cached when running a training that requires a dataset. In order to generate a dataset manually you can run the relative python script in the `daset_scripts` folder. The dataset will be stored in the `data` folder.

## Training the model
```
python main-train.py config.json
```
trains a model and stores the results in the `artifacts/models` folder.

## Constrained generation
```
python main-generate.py config.py
```
Generates samples from a given model and constraint. The folder where the model is located has to be specified in the configuration file.
Some examples of configuration files are already in the config folder.
When experiments are run, a dedicated folder is created with plots and other information in `artifacts/constrained_generation`. 

## Paper plots
In order to reproduce the paper plots, run the dedicated scripts in the `paper_utils` folder:
``` 
python script_name.py experiment_folder
```
The new plots and stats will be stored in the same experiment folder given as input
