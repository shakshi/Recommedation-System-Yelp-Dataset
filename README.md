# Recommedation-System Yelp-Dataset

Built a recommendation system to predict the ratings a user will give to a set of businesses. Yelp dataset is mined to extract interesting and useful information about businesses and users. 

## Dataset
https://www.yelp.com/dataset <br>
Yelp dataset contains restaurants' data and user reviews and ratings for each restaurant. <br>
60% of dataset is used to train the recommedation engine and 40% is used as test data.

## Approach 
The system is a hybrid recommendation system. 
A list of attributes are used to create a feature vector for each restaurant. 
This set of features and previous user-bussiness rating data is used to predict user's rating for a new bussiness (restraunt)

## Features 
Following features are used for each restaurant- 

Price Range (1-4) <br> 
Delivery Service (Yes/ No) <br>
Outdoor Seating (Yes/ No)  <br> 
Good for Groups (Yes/ No) <br> 
Has TV (Yes/No) <br> 
Alcohol (None, Beer & Wine, Full Bar)  <br>
Noise Level (Quiet, Average, Loud, Very Loud)  <br>


## Accuracy 
Performance of the recommendation system on test dataset <br>

Root Mean Square Error: <br> 0.99964 <br>

Error Distribution: <br>
 &gt;=0 and <1: 100354  <br>
 &gt;=1 and <2: 34171 <br>
 &gt;=2 and <3: 6646 <br>
 &gt;=3 and <4: 869 <br>
 &gt;=4 : 4 <br>



