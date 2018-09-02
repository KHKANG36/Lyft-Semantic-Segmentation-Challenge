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
## 1. Data
### Data Acquisition 
1,000 Data is initially provided to each participant. Because 1,000 data is not the enough number for semantic segmentation training, I acquired more data (total 10,860 data) from [CARLA simulator](http://carla.org/) (Gathering more data from CARLA simulator is free for every participant). For data gathering, I came up with two strategies for robust data composition 1) Full diversity - I acquired data under the fully diversified environment such as a variety of weather condition, driving environment, road condition and so on. 2) Maximum number of vehicle & pedestrian - This challenge is the semantic segmentation for two objects, "Road" and "Vehicle". However, in provided data, the areas which "Road" occupy is much larger than those which "Vehicle" does. In this case, the vehicle cannot be detected well after training because road pixel donimate the images. Therefore, the number of vehicle should be over the certain level on the image in order to prohibit the "road data donimation". When I gather the data from CARLA simulator, I set the number of vehicle and pedestrian as maximum to accomplish it. Increased number of pedetrian also help to prohibit the "road data domination" because it helps pedestrian to occupy the road pixel. Here's a sample of the input images (left) and the annotated segmentation images (right) produced by the simulator.
![Test image](https://github.com/KHKANG36/Lyft-Semantic-Segmentation-Challenge/blob/master/data/carla_data/sample_data.png)

### Data Pre-processing
