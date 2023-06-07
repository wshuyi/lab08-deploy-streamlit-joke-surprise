import streamlit as st
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader

@st.cache
def load_data():
    # Load the data
    data_df = pd.read_excel('jester-data/[final] April 2015 to Nov 30 2019 - Transformed Jester Data - .xlsx')
    data_df.columns = range(data_df.shape[1]) # Rename columns to match with jokes dataframe

    # Convert data_df from wide format to long format
    data_df = data_df.stack().reset_index()
    data_df.columns = ['user_id', 'joke_id', 'rating']

    # Filter out missing ratings
    data_df = data_df[data_df['rating'] != 99.0]

    # Load the jokes
    jokes_df = pd.read_excel('jester-data/Dataset4JokeSet.xlsx')
    jokes_df.columns = ['joke']
    jokes_df.index.name = 'joke_id'

    return data_df, jokes_df

def train_model(data_df):
    # Create a Reader object
    reader = Reader(rating_scale=(-10, 10))

    # Load the data into a Dataset object
    data = Dataset.load_from_df(data_df[['user_id', 'joke_id', 'rating']], reader)

    # Build a full trainset from data
    trainset = data.build_full_trainset()

    # Train a SVD model
    algo = SVD()
    algo.fit(trainset)

    return algo

def recommend_jokes(algo, data_df, jokes_df, new_user_id, new_ratings):
    # Convert ratings from 0-5 scale to -10 to 10 scale
    new_ratings = {joke_id: rating*4 - 10 for joke_id, rating in new_ratings.items()}

    # Add new user's ratings to the data
    new_ratings_df = pd.DataFrame({
    'user_id': [new_user_id]*len(new_ratings),
    'joke_id': list(new_ratings.keys()),
    'rating': list(new_ratings.values())
    })

    data_df = pd.concat([data_df, new_ratings_df])

    # Generate recommendations for the new user
    iids = data_df['joke_id'].unique() # Get the list of all joke ids
    iids_new_user = data_df.loc[data_df['user_id'] == new_user_id, 'joke_id'] # Get the list of joke ids rated by the new user
    iids_to_pred = np.setdiff1d(iids, iids_new_user) # Get the list of joke ids the new user has not rated

    # Predict the ratings for all unrated jokes
    testset_new_user = [[new_user_id, iid, 0.] for iid in iids_to_pred]
    predictions = algo.test(testset_new_user)

    # Get the top 5 jokes with highest predicted ratings
    top_5_iids = [pred.iid for pred in sorted(predictions, key=lambda x: x.est, reverse=True)[:5]]
    top_5_jokes = jokes_df.loc[jokes_df.index.isin(top_5_iids), 'joke']

    return top_5_jokes

def main():
    # Load data
    data_df, jokes_df = load_data()

    # Choose an unused user_id for the new user
    new_user_id = data_df['user_id'].max() + 1

    # Randomly display 3 jokes for the user to rate
    st.write('Please rate the following jokes:')
    random_jokes = jokes_df.sample(3)
    new_ratings = {}
    for joke_id, joke in zip(random_jokes.index, random_jokes['joke']):
        st.write(joke)
        rating = st.slider('Rate this joke', 0, 5, step=1)
        new_ratings[joke_id] = rating

    # Add new user's ratings to the data
    new_ratings_df = pd.DataFrame({
    'user_id': [new_user_id]*len(new_ratings),
    'joke_id': list(new_ratings.keys()),
    'rating': [rating*4 - 10 for rating in new_ratings.values()]  # Convert scale from 0-5 to -10-10
    })
    data_df = pd.concat([data_df, new_ratings_df])

    # Train model
    algo = train_model(data_df)

    # Recommend jokes based on user's ratings
    recommended_jokes = recommend_jokes(algo, data_df, jokes_df, new_user_id, new_ratings)

    # Display recommended jokes and ask for user's rating
    st.write('We recommend the following jokes based on your ratings:')
    recommended_ratings = {}
    for joke_id, joke in zip(recommended_jokes.index, recommended_jokes['joke']):
        st.write(joke)
        rating = st.slider('Rate this joke', 0, 5, step=1)
        recommended_ratings[joke_id] = rating

    # Calculate the percentage of total possible score
    total_score = sum(recommended_ratings.values())
    percentage_of_total = (total_score / 25) * 100
    st.write(f'You rated the recommended jokes {percentage_of_total}% of the total possible score.')

if __name__ == '__main__':
    main()
