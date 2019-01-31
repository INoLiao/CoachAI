---
layout: default
title: Coach AI
---

# TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sport Applications
#### <a href="https://github.com/INoLiao" target="_blank">I-No Liao</a>, <a href="https://people.cs.nctu.edu.tw/~yi/" target="_blank">Tsì-Uí İk</a>, <a href="https://sites.google.com/site/wcpeng/" target="_blank">Wen-Chih Peng</a>

## Abstract
<div align="justify">
Ball trajectory data are one of the most fundamental and useful information in the evaluation of players' performance and analysis of game strategies. Although vision-based object tracking techniques have been developed to analyze sports videos, it is still challenging to recognize and position a high-speed and tiny ball from broadcast sport competition videos. In this work, we develop a deep learning network, called TrackNet, to track the badminton from broadcast videos in which the ball images are small, blurry, and sometimes with afterimage tracks or even invisible. The proposed heatmap-based deep learning network is trained to not only recognize the ball image from a single frame but also learn flying patterns from consecutive frames. TrackNet takes images with the size of 640x360 to generate a detection heatmap from several consecutive frames to position the ball and achieve high precision even on public domain videos. The network is evaluated on the video of <a href="https://www.youtube.com/watch?v=__oUhNyM-Jc" target="_blank">2018 Indonesia Open Final - TAI Tzu Ying vs CHEN YuFei</a>. The precision, recall, and F1-measure of TrackNet reach 85.0%, 57.7%, and 68.7%, respectively.
</div>

## Publication
Y.-C. Huang, I.-N. Liao, C.-H. Chen, T.-U. Ik, W.-C. Peng, "TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sport Applications", *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining KDD '19 (Submitted)*

## Dataset
<div align="justify">
Our dataset comes from a video of the badminton competition of <a href="https://www.youtube.com/watch?v=__oUhNyM-Jc" target="_blank">2018 Indonesia Open Final - TAI Tzu Ying vs CHEN YuFei</a>. The resolution is 1280x720 and the frame rate is 30 fps. Unrelated frames such as commercial or highlight replays are screened out. The resulting total number of frames is 18,242. We label each frame with the following attributes: "Frame Name", "Visibility Class", "X", and "Y". Table 1 shows a part of the badminton label file.
</div>

#### Table 1: A part of the badminton label file

| Frame Name | Visibility Class | X   | Y   |
|:----------:|:----------------:|:---:|:---:|
| ...        | ...              | ... | ... |
| 9          | 1                | 693 | 420 |
| 10         | 1                | 692 | 430 |
| 11         | 0                | NaN | NaN |
| 12         | 1                | 701 | 335 |
| ...        | ...              | ... | ... |

<div align="justify">
"Visibility Class" is classified into two categories, VC=0 and VC=1. VC=0 means the ball is not in the frame and VC=1 means the ball is in the frame. "X" and "Y" indicate the coordinate of badminton. In badminton video, prolonged trace often happens due to badminton's fast traveling speed and sometimes we could hardly identify the position of the ball. If the image is prolonged, "X" and "Y" are defined by the latest position of the ball's trace considering its moving direction. An example of how we label the prolonged images is shown in the following figure.
</div>

<br>
<img src="{{ site.baseurl }}/assets/img/ProlongedBadmintonTrace.jpg" alt="Prolonged Badminton Trace" width="640"/>


For badminton dataset download, please click <a href="https://drive.google.com/uc?export=download&id=1ZgoGm5y3_fSwzWLBFe_4Zu4LnMMkUd0J" target="_blank">here</a>.

<div align="justify">
In addition, a video from the tennis men's singles final at the 2017 Summer Universiade is also analyzed for the purpose of model comparison. The resolution, frame rate, and video length are 1280x720, 30 fps, and 75 minutes, respectively. By screening out unrelated frames, 81 game-related clips are segmented and each of them records a complete play, starting from ball serving to score. There are 20,844 frames in total. Each frame possesses the following attributes: "Frame Name", "Visibility Class", "X", "Y". The definition of the attributes is the same as badminton dataset as elaborated previously.
</div>

## TrackNet
<div align="justify">
TrackNet is composed of a convolutional neural network (CNN) followed by a deconvolutional neural network (DeconvNet). It takes consecutive frames to generate a heatmap indicating the position of the object. The number of input frames is a network parameter. One input frame is considered the conventional CNN network. TrackNet with more than one input frame can improve the moving object tracking by learning the trajectory pattern. TrackNet is trained to generate a probability-like detection heatmap having the same resolution as input frames. The ground truth of the heatmap is an amplified 2D Gaussian distribution located at the center of the badminton. The coordinates of the ball are available in the labeled dataset and the variance of the Gaussian distribution refers to the diameter of badminton images. The implementation details of TrackNet is illustrated in the following figure. The input of the proposed network can be some number of consecutive frames. The first 13 layers refer to the design of the first 13 layers of VGG-16 [1] for object classification. The 14-24 layers refer to DeconvNet for semantic segmentation [2]. To realize the pixel-wise prediction, upsampling is applied to recover the information loss from maximum pooling layers. Symmetric numbers of upsampling layers and maximum pooling layers are implemented.
</div>

<br>
![alt text]({{ site.baseurl }}/assets/img/TrackNet.png "TrackNet")

## Experimental Results
<div align="justify">
Before the evaluation of the badminton dataset, our previous experimental results on tennis tracking is introduced. The tennis dataset comes from the video of the men's singles final at the 2017 Summer Universiade. Both Archana's algorithm [3], a conventional image processing technique, and the proposed TrackNet are evaluated. The dataset contains 20,844 frames and is randomly divided to the training set and test set. 70% frames are the training set and 30% frames are the test set. To speed up the training speed, all frames are resized from 1280x720 to 640x360. To compare the performance of TrackNet frameworks with one single input frame and three consecutive input frames, two versions of TrackNet are implemented. For convenience, TrackNet that takes single input frame is named as Model I and TrackNet that takes three consecutive input frames is named as Model II. Note that TrackNet framework is scalable. Any number of consecutive input frames are allowed. Table 2 shows the model training parameters, including learning rate, batch size, steps per epoch, number of epochs, etc.
</div>

#### Table 2: Parameters used in model training

| Parameters                | Setting           |
|:-------------------------:|:-----------------:|
| Learning rate             | 1.0               |
| Batch size                | 1                 |
| Steps per epoch           | 200               |
| Epochs                    | 500               |
| Initial weights           | random uniform    |
| Range of initial weights  | [-0.05, 0.05]     |

<div align="justify">
The overall performance in terms of precision, recall, and F1-measure are summarized in Table 3. It is observed that compared to Archana's algorithm [3], the performance is significantly improved for both TrackNet Model I and TrackNet Model II. The comparison presents an exceptional object detection capability of deep learning networks over conventional image processing algorithms. In addition, TrackNet Model II performs even better than TrackNet Model I, proving that training TrackNet with consecutive input frames can further improve its dynamic object tracking ability, especially for small objects. This discovery directly exhibits that consecutive frames provide critical information for the network to learn trajectory patterns of the interested object.
</div>

#### Table 3: Performance of tennis tracking

| Model             | Precision | Recall | F1-measure |
|:-----------------:|:---------:|:------:|:----------:|
| Archana's [3]     | 92.5%     | 74.5%  | 82.5%      |
| TrackNet Model I  | 95.7%     | 89.6%  | 92.5%      |
| TrackNet Model II | 99.8%     | 96.6%  | 98.2%      |

<div align="justify">
As for badminton, the badminton dataset contains 18,242 frames with the resolution of 1280x720. Similarly, all frames are resized from 1280x720 to 640x360 to speed up the training process. The dataset is randomly divided to the training set and test set. 70% frames are the training set and 30% frames are the test set. The model training parameters are set to the same values used in the training of tennis dataset as shown in Table 2. To verify the feasibility of TrackNet framework on badminton tracking, we train a model named as TrackNet-Badminton which is trained by badminton dataset using three consecutive input frames. As shown in Table 4, TrackNet-Badminton reaches precision, recall, and F1-measure of 85.0%, 57.7%, and 68.7%, respectively.
</div>

#### Table 4: Performance of badminton tracking

| Model              | Precision | Recall | F1-measure |
|:------------------:|:---------:|:------:|:----------:|
| TrackNet-Badminton | 85.0%     | 57.7%  | 68.7%      |

<div align="justify">
Compared with tennis tracking, it can be observed that tennis tracking outperforms badminton tracking by a noticeable margin. This is because badminton travels much faster than tennis, resulting in much more unclear object images in badminton videos. The fastest serve according to the official records from the Association of Tennis Professionals is John Isner's 253 kilometers per hour at the 2016 Davis Cup. On the other hand, the fastest badminton hit in competition is Lee Chong Wei's 417 kilometers per hour smash at the 2017 Japan Open according to Guinness World Records, which is over 1.6 times faster than tennis. In fact, in professional competitions, the speed of badminton is frequently over 300 kilometers per hour. Such an enormous increase in velocity causes performance degradation especially in the aspect of the recall due to high false negatives. High traveling speed makes the badminton move across long distance within only a few frames. The property of dynamic trajectories in such high speed becomes hard to recognize by the model. In addition to the absolute speed, badminton possesses a much higher variation in traveling speed than tennis. For example, in badminton, a drop stroke and a smash stroke have a significant difference in velocity. Such extreme scenarios commonly happen during a badminton competition, making the model hard to fit both scenarios perfectly. Nonetheless, although the performance in tracking badminton is not as phenomenal as tennis, achieving a precision of 85.0% is accurate enough to correctly depict all trajectories in the game. Future research on TrackNet improvement in the aspects of identifying trajectories of extreme fast objects and learning distinct patterns caused by significant speed variation will be conducted.
</div>

#### Reference: 

[1] Karen Simonyan, and Andrew isserman. 2014. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556 (2014).<br>
[2] Hyeonwoo Noh, Seunghoon Hong, and Bohyung Han. 2015. Learning deconvolution network for semantic segmentation. In Proceedings of the IEEE International Conference on Computer Vision. 1520–1528.<br>
[3] M. Archana and M. Kalaisevi Geetha. 2015. Object detection and tracking based on trajectory in broadcast tennis video. Procedia Computer Science 58 (2015), 225–232.

## Demo Video

#### Badminton tracking by TrackNet in the competition of 2018 Indonesia Open Final - TAI Tzu Ying vs CHEN YuFei:

<iframe width="640" height="360" src="https://www.youtube.com/embed/62tLJvLlAA0" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>