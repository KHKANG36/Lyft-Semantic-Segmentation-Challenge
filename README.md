# Lyft Challenge
### Semantic Segmentation for Vehicle and Road in Video ___(Finally Ranked on 38th out of 155)___
![Test image](https://github.com/KHKANG36/Lyft-Semantic-Segmentation-Challenge/blob/master/data/challenge_result/Main.gif)


### Final Ranking and Score (The last day of Challenge) ___Ranked 38th, Total Score : 88.2837___

![Test image](https://github.com/KHKANG36/Lyft-Semantic-Segmentation-Challenge/blob/master/data/challenge_result/Score.PNG)


### Challege Goal 

- Get the highest score for pixel-wise segmentation of cars and road surface from the camera data inside a simulated car.
 
### Scoring/Ranking Criteria 
Weighted F-Score
The challenge grades and ranks participants on a weighted F-score.
It's more important to have better precision on detecting the road surface and better recall on cars (because you'd rather avoid a non-existent car (false positive) than crash into one you didn't detect (false negative)).

![Test image](https://github.com/KHKANG36/Lyft-Semantic-Segmentation-Challenge/blob/master/data/challenge_result/Fscore.png)

It was chosen that the beta for cars would be 2 and beta for roads would be 0.5.
After reviewing the video segmentation samples from the model, the car precision is still unacceptably poor. In my video outputs, there were many false positive for cars, which could lead to a jerky rider experience for a real self driving car.
I use a weighted loss function and different probability thresholds for the cars vs roads. Furthermore, when the output (binary mask) of the network is resized to fit the frame of the video, I made sure that the car masks included more of the fringe artifacts of resizing than the roads.
