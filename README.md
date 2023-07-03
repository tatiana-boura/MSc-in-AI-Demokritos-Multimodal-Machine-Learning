# Video Clip Retrieval
#### MSc-in-AI-Demokritos-Multimodal-Machine-Learning 
------------------------------------------------
### Authors
| Name | Registration number |
| ------ | ------ |
| Boura Tatiana | MTN2210 |
| Sideras Andreas | MTN2214 |


## Contents of this repository: 
- [Report.pdf] - The report of our project.
- [Presentation.pdf] - The presentation of our project.
  
### Process
Run the process by executing the following scripts or notebooks,
-    To extract features from all 3 modalities execute the function ```get_features(test=False)``` from  [feature extraction/get_features.py].
-    To extract the embeddings of the features using the *AutoEncoder*, execute the notebook [representation_learning.ipynb].
-    To see the experiments or execute your own, execute the notebook [video_queries.ipynb].

To execute the experiment you just specify the YouTube title of the video-clip/query and it is automatically downloaded in the folder ```../data/data_test```.

**Note** : To run your own experiment make sure to create the folder ```../data/data_test```, where the root is the current repository. This folder should be empty every time you add a query.

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

[Presentation.pdf]: 
<https://github.com/tatiana-boura/MSc-in-AI-Demokritos-Multimodal-Machine-Learning/blob/main/multimodal_presentation.pdf>
[Report.pdf]: 
<https://github.com/tatiana-boura/MSc-in-AI-Demokritos-Multimodal-Machine-Learning/blob/main/Multimodal_Report.pdf>
[feature extraction/get_features.py]: 
<https://github.com/tatiana-boura/MSc-in-AI-Demokritos-Multimodal-Machine-Learning/blob/main/feature%20extraction/get_features.py>
[representation_learning.ipynb]:
<https://github.com/tatiana-boura/MSc-in-AI-Demokritos-Multimodal-Machine-Learning/blob/main/representation_learning.ipynb>
[video_queries.ipynb]:
<https://github.com/tatiana-boura/MSc-in-AI-Demokritos-Multimodal-Machine-Learning/blob/main/feature%20extraction/video_queries.ipynb>
