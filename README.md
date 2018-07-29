# Lyft Challenge
### Semantic Segmentation for Vehicle and Road in Video ___(Finally Ranked on 38th out of 155)___
![Test image](https://github.com/KHKANG36/Lyft-Semantic-Segmentation-Challenge/blob/master/data/challenge_result/Main.gif)


### Final Ranking and Score (The last day of Challenge) ___Ranked 38th, Total Score : 88.2837___

![Test image](https://github.com/KHKANG36/Lyft-Semantic-Segmentation-Challenge/blob/master/data/challenge_result/Score.PNG)


### Challege Goal 

- Get the highest score for pixel-wise segmentation of cars and road surface from the camera data inside a simulated car.
 
### Scoring/Ranking Criteria 

Weighted F-Score
The challenge grades and ranks participants on a weighted F-score.It's more important to have better precision on detecting the road surface and better recall on cars (because you'd rather avoid a non-existent car (false positive) than crash into one you didn't detect (false negative)).

![Test image](https://github.com/KHKANG36/Lyft-Semantic-Segmentation-Challenge/blob/master/data/challenge_result/Fscore.png)

It was chosen that the beta for cars would be 2 and beta for roads would be 0.5 in this challenge.

## My Approach
### 1) Data
![Test image](https://github.com/KHKANG36/Lyft-Semantic-Segmentation-Challenge/blob/master/data/carla_data/sample_data.png)
All camera data for this project comes from the the 'CARLA' simulator. Gathering data from a simulator is much faster than the real-world and allows us to iterate on our perception pipeline quickly before fine tuning it with real world data.
![Test image](https://github.com/KHKANG36/Lyft-Semantic-Segmentation-Challenge/blob/master/data/carla_data/10026.png)
