# Image Classification based on transfer learning and Siamese Network

The goal of transfer learning is to improve the performance of a model on target domains by transferring the knowledge learned from different but related source domains. Siamese network is a set of neural network architectures that contains two or more subnet- works with the same configuration which share parameters with each other. A subnetwork is typically a CNN model, and all the sub- networks are connected by a loss function. It is normally used to measure the similarity between the input images. This project aims to compare using simple Siamese network features with transfer learning approach on shoes dataset classification accuracy. Also, investigate the effect of the Siamese network on the features orig- inated from transfer learning. The results show that the features from our implementation of the Siamese network lead to better ac- curacy. Farther more, it shows a significant improvement in the feature from transfer learning.

## Transfer Learning 

<img src="https://github.com/KokoFan16/DeepLearningTeamProject2021/blob/main/report/figs/transfer_learning.png" width="700"/>

<img src="https://github.com/KokoFan16/DeepLearningTeamProject2021/blob/main/report/figs/xception.png" width="700"/>

## Siamese Network

In this project, we implement both two loss functions with the architecture shown in the figure below.
<img src="https://github.com/KokoFan16/DeepLearningTeamProject2021/blob/main/report/figs/model.png" width="800"/>

This is the training process with contrastive loss function.
<img src="https://github.com/KokoFan16/DeepLearningTeamProject2021/blob/main/report/figs/contrastive.png" width="700"/>

This is the training process with triplets loss function.
<img src="https://github.com/KokoFan16/DeepLearningTeamProject2021/blob/main/report/figs/triplet.png" width="700"/>


## Dataset

UT Zappos50K (UT-Zap50K) is a large shoe dataset consisting of 50,025 catalog images collected from Zappos.com. The images are divided into 4 major categories — shoes, sandals, slippers, and boots — followed by functional types and individual brands. The shoes are centered on a white background and pictured in the same orientation for convenient analysis. The numbers of images of four categories are ```[('Slippers', 1283), ('Sandals', 5741), ('Boots', 12832), ('Shoes', 30169)]```.

Download Dataset
```
!wget http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images.zip
!unzip ut-zap50k-images.zip
```

Examples of four classes in the dataset.

<img src="https://github.com/KokoFan16/DeepLearningTeamProject2021/blob/main/report/figs/data_example.png" width="600"/>

In order to create a balanced dataset, we use a subset of this dataset, which contains 1283 slippers, 1500 sandals, 1500 boots and 1500 shoes.

## Structure
All the Jupyter-notebook files are in the ```src``` folder. Our presentation slide is in the ```ppt``` folder. And the latex-based report is in the ```report``` folder. 

In the ```src``` folder, we have five files, and they are for different implementations. You can easily differentiate them by the file name.  

## Execution
These commands are only for the UAB Cheaha server (https://docs.uabgrid.uab.edu/wiki/cheaha). <br>
### Connect to Cheaha 
Start your terminal. <br>
Copy/Paste this command in your local terminal to connect to Cheaha. <br>
```
ssh blazerid@cheaha.rc.uab.edu
```

### Clone CNER
Copy/Paste this command in your local terminal to clone this project:
```
git clone https://github.com/uabinf/nlp-group-project-fall-2020-cner.git
```

### UAB Cheaha Sever
There are two methods to run jupyter notebook on UAB Cheaha Sever. <br>
You can choose one of them.<br>
#### 1. Cheaha Dashboard 
(https://rc.uab.edu/pun/sys/dashboard)
```
Interactive Apps -> Jupyter Notebook
```
Then you can see the page as below: <br>
<img src="https://github.com/uabinf/nlp-group-project-fall-2020-cner/blob/main/image/cheaha.png" width="1000"/>
```
Set the parameters -> Launch -> Connect to Jupyter

Parameters: 
module load cuda10.0/toolkit
module load Anaconda3
--notebook-dir=/data/user/$USER/nbotw  --gres=gpu:1
12
pascalnodes
8
16
```
Copy/Paste this in your Cheaha terminal to ssh to your applied host: <br>
```
ssh c<host_id>
```

#### 2. Cheaha Terminal
Copy/Paste this in your Cheaha terminal to submitted batch job:
```
sbatch ./script/cheaha_job.sh
```
You will get a log file `jupyter-log-pascal-<job_id>.txt`. <br>
Here is an exmaple of the log file: <br>
```
The following have been reloaded with a version change:
  1) CUDA/9.2.148.1 => CUDA/10.1.243



   Copy/Paste this in your local terminal to ssh tunnel with remote  
   ------------------------------------------------------------------
   ssh -L 8349:172.20.201.103:8349 <blazerid>@cheaha.rc.uab.edu           
   ------------------------------------------------------------------


   Then open a browser on your local machine to the following address
   ------------------------------------------------------------------
   localhost:8349                                                
   ------------------------------------------------------------------


/share/apps/rc/software/Anaconda3/5.3.1/lib/python3.7/site-packages/notebook/services/kernels/kernelmanager.py:19: VisibleDeprecationWarning: zmq.eventloop.minitornado is deprecated in pyzmq 14.0 and will be removed.
    Install tornado itself to use zmq with the tornado IOLoop.
    
  from jupyter_client.session import Session
[I 14:35:43.631 NotebookApp] [nb_conda_kernels] enabled, 5 kernels found
[I 14:35:48.993 NotebookApp] JupyterLab extension loaded from /share/apps/rc/software/Anaconda3/5.3.1/lib/python3.7/site-packages/jupyterlab
[I 14:35:48.993 NotebookApp] JupyterLab application directory is /data/rc/apps/rc/software/Anaconda3/5.3.1/share/jupyter/lab
[I 14:35:49.088 NotebookApp] [nb_conda] enabled
[I 14:35:49.088 NotebookApp] Serving notebooks from local directory: /data/user/<blazerid>
[I 14:35:49.088 NotebookApp] The Jupyter Notebook is running at:
[I 14:35:49.088 NotebookApp] http://172.20.201.103:8349/?token=fab72652490fb10b50e798716914dba5127ebde6ec4660f2
[I 14:35:49.089 NotebookApp]  or http://127.0.0.1:8349/?token=fab72652490fb10b50e798716914dba5127ebde6ec4660f2
[I 14:35:49.089 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 14:35:49.126 NotebookApp] 
```

Copy/Paste this in your Cheaha terminal:
```
ssh -L <ip adddress> <your username>@cheaha.rc.uab.edu
```
Copy/Paste this in brower to open jupyter notebook:
```
[I 16:30:27.170 NotebookApp] The Jupyter Notebook is running at:
[I 16:30:27.170 NotebookApp] http://<ip address>/?token=<token_id>
```

### Enviroment setup
Please do not execute the following commands on the Login Node! You are supposed to request a resource from Cheaha, and 'ssh' to the resource nodes!

Copy/Paste this command in your local terminal to load Anaconda module:
```
module load Anaconda3/2020.07
```
Copy/Paste this command in your local terminal to create Environment:
```
conda env create -f environment.yml
```
Copy/Paste this command in your local terminal to check your environment list, after create environment sucessfully. The environment called ```project``` is the environment for this project.
```
conda env list
```

Please use the installed kernal ```dlproject``` to run the project.


## Deadline
1. Term Project Presentation -- Apr 22 by 12pm
2. Term Project Report and Code -- Apr 24 by 11:59pm

