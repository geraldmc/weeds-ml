
## DL4CV - Image generation using DCGAN and ACGAN

### Requirements

The project code has been developed to run either locally or on Google Colab as a Jupyter notebook. If running Colab, no dependencies beyond what Colab supplies are required. It is recommended that the Colab runtime be set to 'GPU' and 'High Ram' but it will execute fine if these are unset. If a library is missing it may be installed with the following:

```bash

!pip install <library>

```

### Data
The required images for running the code are available in zip format here: 

* [images.zip](https://drive.google.com/file/d/1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj) (468 MB)
  
The image bundle must be downloaded and extracted into a directory named `images` at the project root.

### Data organization
Images are assigned unique filenames that include the date/time the image was photographed and an ID number. The format is: ```YYYYMMDD-HHMMSS-ID```, where ID is an integer from 0 to 3. Unique filenames are strings of 17 characters, such as 20170320-093423-1.

A labels.csv file with path references to all files is available as a csv in the format:

```
Filename,Label,Species
...
20170207-154924-0,jpg,7,Snake weed
20170610-123859-1.jpg,1,Lantana
20180119-105722-1.jpg,8,Negative
```

K-folds training (with k=5) is applied during train, validation and testing. Train, val and test sets for k-folds have been assigned individual csv files corresponding to each fold of that run, i.e.:

```
train_subset0.csv
train_subset1.csv
...
...
val_subset0.csv
val_subset1.csv
...
...
test_subset0.csv
test_subset1.csv
```

## Notebooks
### DeepWeeds Data Loading (local Machine)*

i) `load_deepweeds_local.ipynb`: used for loading the DeepWeeds dataset into Pytorch's dataload format at the `data/train/[class label]/`, `data/val/[class label]/` and `data/val/[class label]/` directories.

ii) `DataAugmentation.ipynb`: used for augmenting the training data. Augmented training data is saved in `data/train_dataaug` folder. Please follow https://imgaug.readthedocs.io/en/latest/ to install library.

iii) `Generate_ACGAN_Images.ipynb`: used to generate ACGAN images and save them in the `data/test_generated/[class label]` folder.

#### DCGAN
i) `DCGAN.ipynb`: used for training the generator/discriminator. This notebook requires that the data file be in the same directory as the notebook. The images should be stored as `data/train/[class label]/` directory structure. The notebook will also save the generator model in `Model` folder.

_Dependency for DCGAN.ipynb_: `dcgan_network.py` in the same directory.

ii) `DCGAN Evaluation.ipynb`: used for evaluating the DCGAN generator model. This notebook requires a trained generator model file `Model/deepweeds_DCGAN_gan_0.pth` to exist in the `Model` folder.

#### ACGAN

i) `AC-GAN.ipynb`: used to train the generator/discriminator. This notebook requires a data file to be in the same directory as the notebook. The images should be stored using a `data/train/[class label]/` directory structure. In order to train the model on augmented data, first run the `DataAugmentation.ipynb`. The train data for augmented data is located in `data/train_dataaug/[class label]/`. The notebook will also save the generator model in `Model` folder.

_Dependency for AC-GAN.ipynb_:  `acgan_network.py` in the same directory.

  

ii) `ACGAN Evaluation.ipynb`: used for evaluating the ACGAN generator model. The notebook will require a trained generator model file `Model/deepweeds_acgan_gan_v3.pth` to exist in the model folder. The `AC-GAN.ipynb` must be run first if the trained model does not exist.

### DeepWeeds Train, Eval, Test (Colab)*


i) `main.ipynb`: this notebook is used to alter, train, evaluate, and test a Pytorch ResNet-50 model for k-folds. The notebook was developed to be run on Colab and includes dependencies specific to that environment.


#### Data dependency

Steps required to setup the runtime environment on Colab are as follows:

 - Download the code from Github (run without change)
 - Import the project code (run without change).    
 - Download and unzip the dataset from Drive

**Note**: The third step is specific to user and environment as it depends on accessing Google Drive. The file `params.py` under the project code directory `conf` will have to be customized.


