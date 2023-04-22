import pickle
import random
import numpy as np
import streamlit as st
from PIL import Image
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from deepface import DeepFace


   
# Define a function to convert the image to a numpy array
# def img_to_array(img):
#     img = np.array(img.convert('RGB'))
#     return img[:, :, ::-1].copy()
   
st.title("Welcome to Emotion-based music recommender")
st.write("How it works")
st.write("Kindly input your emotion so we can give you some recommendations")
# def emotion():
    # try:
    #     img = st.camera_input("Take a picture")
    #     if img:
    #         img = Image.open(img)
    #         img_array = img_to_array(img)
    #         result = DeepFace.analyze(img_array, actions=['emotion'])
    #         emotion = result[0]['dominant_emotion']
    #         st.write("Your emotion based on the image is: ", emotion)
    #         return emotion
    # except ValueError:
        # st.write("Sorry, a face could not be detected")
emotion = st.text_input("Enter an emotion: ").lower()
st.write("Dominant emotion detected: ", emotion)
    # return emotion
    
with open('model.sav', 'rb') as f:
    kmeans, sad_dataset, happy_dataset, neutral_dataset = pickle.load(f)

kmeans = kmeans
sad_dataset = sad_dataset
happy_dataset = happy_dataset
neutral_dataset = neutral_dataset
dominant_emotion = emotion




#The below function is to get the dataset when an emotion is detected.
def get_data():
    # Check if dominant emotion is happy
    if dominant_emotion == 'happy':
        # Return happy dataset if emotion is happy
        return happy_dataset
    
    # Check if dominant emotion is sad
    elif dominant_emotion == "sad":
        # Return sad dataset if emotion is sad
        return sad_dataset
    
    # If dominant emotion is neutral or not detected
    else:
        # Return neutral dataset as default
        return neutral_dataset

# Initialize empty lists for songs of different moods
sad_songs = []
happy_songs = []
other_songs = []

# Create a list of the three mood-specific lists
list_of_lists = [sad_songs, happy_songs, other_songs]

# Initialize a counter variable
point = 0

# Function to create a list of songs of a particular mood
def create_list(mood, data=get_data()):
    global point
    
    # Loop through the rows of the dataset
    for row in data.itertuples():
        # If mood is happy, add a random happy song to the happy list
        if mood == 'happy':
            sample_song = data.sample()
            happy_songs.append(sample_song['name'].to_string(index=False))
            point = point + 1
            if point == 5:
                break
                
        # If mood is sad, add a random sad song to the sad list
        elif mood == 'sad':
            sample_song = data.sample()
            sad_songs.append(sample_song['name'].to_string(index=False))
            point = point + 1
            if point == 5:
                break
                    
        # If mood is neither happy nor sad, add a random song to the other list
        else:
            sample_song = data.sample()
            other_songs.append(sample_song['name'].to_string(index=False))
            point = point + 1
            if point == 5:
                break
                        
    return

create_list(dominant_emotion)



# Initialize a CountVectorizer object
vectorizer = CountVectorizer()

# Use the fit() method to train the vectorizer on the data obtained from get_data() function
# and store it in the vectorizer variable
vectorizer = vectorizer.fit(get_data())


# function to calculate similarities between input song and all other songs in the dataset
#This function is to get the similarity of the input songs with other songs in the dataset
def get_similarities(name,data):
    # Getting vector for the input song.
    text_array1 = vectorizer.transform(data[data['name']==name]).toarray()
    num_array1 = data[data['name']==name].select_dtypes(include=np.number).to_numpy()
    
    # We will store similarity for each row of the dataset.
    sim = []
    for idx, row in data.iterrows():
        song_name = row['name']
        
        # Getting vector for current song.
        text_array2 = vectorizer.transform(data[data['name']==song_name]).toarray()
        num_array2 = data[data['name']==song_name].select_dtypes(include=np.number).to_numpy()
        
        # Calculating similarities for text as well as numeric features
        text_sim = cosine_similarity(text_array1, text_array2)[0][0]
        num_sim = cosine_similarity(num_array1, num_array2)[0][0]
        sim.append(text_sim + num_sim)
        
    return sim

# This function recommends songs based on the similarity factor and popularity
def recommend_songs(name, data):
    # Calculate similarity factor
    data['similarity_factor'] = get_similarities(name, data)
    
    # Sort by similarity factor and release date in descending order
    data.sort_values(by=['similarity_factor', 'year'], ascending=[False, False], inplace=True)
    
    # Remove duplicates of the input song
    data.drop_duplicates(subset=['name'], inplace=True)
    data = data.reset_index(drop=True)

    
    # The first song will be the input song itself as the similarity will be highest
    recommended_songs = data.loc[:, ['name', 'artist']].iloc[1:11]
    
    return recommended_songs



# This function recommends 10 songs similar to a random song chosen from a list of sad, happy, and other songs.
def final_recommendation(data=get_data()):
    # Iterate over the list of song moods and choose a random song from each list
    for mood_list in list_of_lists:
        if len(mood_list) != 0:
            song_choice = random.choice(mood_list)
            # st.write('The recommended song based on your mood is',song_choice)
            recommended_songs = recommend_songs(song_choice, data)
            # print(recommended_songs.to_string(index=False))
            # st.write(recommended_songs)

    return recommended_songs, song_choice

x = final_recommendation(get_data())
st.write("The recommended song based on your mood is", x[1])
st.write(x[0])
