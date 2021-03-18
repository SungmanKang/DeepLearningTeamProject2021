# DeepLearningTeamProject2021

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
