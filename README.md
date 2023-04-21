# Music Recommendation System
---
Author: Beauty Ikudehinbu

This project contains a machine learning model that recommends songs based on the user's emotion. The emotion is detected using a pre-trained model called deepface. The model then uses k-means clustering to determine which songs are happy, sad, or neutral based on their variance.

## Files
- Song_recommendation.ipynb: This file contains the code for the machine learning model used for song recommendation. The code includes the pre-trained model, deepface, and k-means clustering. Note that the code also includes an implementation of DBSCAN, which has been commented out due to memory issues.

- app.py: This is the file for the music recommendation app built using Streamlit. The app handles exceptions in cases where the user's face cannot be detected. In such cases, the user can input their emotion, and the recommended song is displayed in a tabular form.

- data.csv: This is the music data used for the project. It was obtained from Kaggle.

- model.sav: This is the trained machine learning model deployed in the app.py file.

- requirements.txt: This file contains the required functions and libraries used in the project.

## Instructions
To run the music recommendation system, follow these instructions:

* Clone this repository to your local machine.
* Install the required libraries by running pip install -r requirements.txt in your terminal.
* Run streamlit run app.py in your terminal to launch the app.
* Click on the "Start" button to detect your emotion.
* If your emotion cannot be detected, input your emotion manually.
* Click on the "Recommend Song" button to view the recommended song.
Thank you for using the music recommendation system!

## Walkthrough Video
https://clipchamp.com/watch/W6QeRpLr0Vz

## Deployed 
https://mzbhewtee-music-recommendation-app-aan631.streamlit.app/ 



