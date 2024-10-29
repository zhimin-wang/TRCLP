## Tasks Reflected in the Eyes: Egocentric Gaze-Aware Visual Task Type Recognition in Virtual Reality

Project homepage: https://zhimin-wang.github.io/TaskTypeRecognition.html
The code of this project is modified on [EHTask](https://github.com/CraneHzm/EHTask) framework, thank Zhiming Hu for enthusiastically solving our doubts for many times.
![](README_md_files/326903e0-959a-11ef-945f-897d79f97fb8.jpeg?v=1&type=image)

### Introduction
With eye tracking finding widespread utility in augmented reality and virtual reality headsets, eye gaze has the potential to recognize users’ visual tasks and adaptively adjust virtual content displays, thereby enhancing the intelligence of these headsets. However, current studies on visual task recognition often focus on scene-specific tasks, like copying tasks for office environments, which lack applicability to new scenarios, e.g., museums. In this paper, we propose four scene-agnostic task types for facilitating task type recognition across a broader range of scenarios. We present a new dataset that includes eye and head movement data recorded from 20 participants while they engaged in four task types across 15 360-degree VR videos. Using this dataset, we propose an egocentric gaze-aware task type recognition method, TRCLP, which achieves promising results. Additionally, we illustrate the practical applications of task type recognition with three examples. Our work offers valuable insights for content developers in designing task-aware intelligent applications.

### Environment
Please refer to requirements.txt or run:
```
pip install -r requirements.txt
```
### Usage
#### Step1: Prepare data
You can directly download dataset from [here](https://drive.google.com/file/d/1HW-MxPx6v0HxBBAq_IATj5eouSvG3daU/view?usp=sharing), and place it in a location parallel to /TRCLP as shown below. Then process the dataset according to the /ReadMe.txt.
```
/TRCLP
/Dataset
```
#### Step2: Test on the pretrained model
You can test the pretrained model for evaluating the accuracy by:
```
. run_pretrained_cross_user_models 
```
or 
```
. run_pretrained_cross_scene_models
```
If you want to check the specific example, please modify ``--onlyFilter 1`` to ``--onlyFilter 0``, and run above .sh. Then check `/predictions/cross_user/Test_Fold_1/` and you will find the output as shown below. This figure shows the specific tasks that users are performing. The 360-degree video can also be found in this  dataset.
![](README_md_files/465905c0-95a0-11ef-945f-897d79f97fb8.jpeg?v=1&type=image)

#### Step3: Retrain our model
You can retrain our model for the five-fold test by:
```
. training_cross_user_models
```
or 
```
. training_cross_scene_models
```
Above results don't include the optimized results. Please check ``/predictions/cross_user/Test_Fold_X/`` to find the ``epoch_15_test_visualization.jpg`` and ``epoch_30_test_visualization.jpg``. We can use the best retrained model to test, such Encoder-15, Classifer-16 for Fold_01. Then we can modify the Fold-01 commands ``--runEncoder 30 --runClassifer 28`` to ``--runEncoder 15 --runClassifer 16`` in ``run_pretrained_cross_user_models ``  for using the post-optimization.
![](README_md_files/8da5ab20-95a2-11ef-945f-897d79f97fb8.jpeg?v=1&type=image)

### Future work
We plan to open source the data collection system and the eye movement trajectory visualization system in Unity3D, similar to the following picture. However, this workload is extremely huge. I don't have the time to do this now. If you have relevant needs, please let me know and I will adjust according to the situation. Email: zm.wang@buaa.edu.cn.
![](README_md_files/97a36350-95a3-11ef-945f-897d79f97fb8.jpeg?v=1&type=image)

### Citations
```
@ARTICLE{Wang_TVCG2024A,
    author={Wang, Zhimin and Lu, Feng},
    journal={IEEE Transactions on Visualization and Computer Graphics}, 
    title={Tasks Reflected in the Eyes: Egocentric Gaze-Aware Visual Task Type Recognition in Virtual Reality}, 
    year={2024},
    volume={30},
    number={11},
    pages={7277-7287},
    keywords={Visualization;Switches;Extended reality;Measurement;Tracking;Annotations;Gaze tracking;Virtual reality;eye tracking;visual task type recognition;deep learning;intelligent application},
    doi={10.1109/TVCG.2024.3456164}}
```
