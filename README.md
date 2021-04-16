# DeepLearningTeamProject2021

## Dataset

UT Zappos50K (UT-Zap50K) is a large shoe dataset consisting of 50,025 catalog images collected from Zappos.com. The images are divided into 4 major categories — shoes, sandals, slippers, and boots — followed by functional types and individual brands. The shoes are centered on a white background and pictured in the same orientation for convenient analysis. The numbers of images of four categories are ```[('Slippers', 1283), ('Sandals', 5741), ('Boots', 12832), ('Shoes', 30169)]```.

Download Dataset
```
!wget http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images.zip
!unzip ut-zap50k-images.zip
```

Examples of four classes in the dataset.

<img src="https://github.com/KokoFan16/DeepLearningTeamProject2021/blob/main/images/data_example.png" width="600"/>

In order to create a balanced dataset, we use a subset of this dataset, which contains 1283 slippers, 1500 sandals, 1500 boots and 1500 shoes.


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

## Ideas

1. Social Distancing Detector: https://www.analyticsvidhya.com/blog/2020/05/social-distancing-detection-tool-deep-learning/
2. Perform transfer learning to learn a new set of object categories: https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/ and https://keras.io/api/applications/ (to download images: https://pypi.org/project/bing-image-downloader/)  (to perform transfer learning for object detection http://www.vision.caltech.edu/pmoreels/Datasets/Home_Objects_06/)

## Requirements

The following four types of projects are acceptable, listed in the increasing order of difficulty. Bottom line - there is something new (‘new’ does not necessarily mean ‘novel’) in your project that has not been done before.

1. A new application: pick a new application and explore how best apply DL algorithms (existing or adapted) to solve this new problem. One low hanging fruit could be to perform transfer learning to learn a new set of object categories that are not already in those popular image datasets https://lionbridge.ai/datasets/20-best-image-datasets-for-computer-vision/

20 Free Image Datasets for Computer Vision | Lionbridge AI
Where’s the best place to look for free online datasets for image tagging? We combed the web to create the ultimate cheat sheet of open-source image datasets for machine learning.
lionbridge.ai

Another idea that just jump into my mind is the detection of small or big crowd gathering that do not follow the social distancing.

This is the easiest category so make sure you pick an interesting application that really excites you.

2. A new learning algorithm or a novel variant of an existing algorithm, to better solve an existing problem. An easy target could be to develop a DL algorithm for an old problem that had been tackled by traditional ML techniques in the past but does not have a DL update. another example (more difficult) could be to better design the regularizer in the loss function so that learning becomes better and faster - one such example is the use of a ’penalty’ regularizer for the non-central image areas in an image object classification application.

And/Or, injecting and integrating additional data/information source to better solve an existing problem, i.e., novel data augmentation and info. integration techniques. These do not include the commonly used augmentation techniques such as duplicating, image flipping, cropping, or rotation.

3. Replicate the framework and results in a recent research paper published in renowned venues such as CVPR, ICCV, AAAI, NISP, and ACM Multimedia, provided that the code of the paper is not disclosed. If the original dataset cannot be obtained, apply to a smaller, self collected dataset that have the same or similar nature as the original one .

4. Purely theoretical work on topics such as better general architectures for computer vision and NLP, systematic and efficient hyper parameter tuning, and interpretability of DL.

Evaluation is based on both the project presentation and the final report. Factors include technical soundness, novelty (problem and/or the approach), experiments (how does it compare to other ML approaches (if the same problem)? Or, how does it compare to traditional ML approaches if a new problem?), and discussions (i.e., reasons for success and failure cases and future improvements.)
