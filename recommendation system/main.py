import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

@st.cache_data
def load_data():
    # Load the ratings data
    df = pd.read_csv('u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    return df

@st.cache_resource
def build_user_item_matrix(df):
    user_item = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    return user_item

@st.cache_resource
def compute_user_similarity(user_item):
    similarity = cosine_similarity(user_item)
    sim_df = pd.DataFrame(similarity, index=user_item.index, columns=user_item.index)
    return sim_df

def recommend_for_user(user_id, user_item, sim_df, top_n=10):
    if user_id not in user_item.index:
        return None

    # Get similarity scores for the target user
    sim_scores = sim_df[user_id]

    # Weighted sum of ratings from other users
    user_ratings = user_item.loc[user_id]
    not_rated_items = user_ratings[user_ratings == 0].index

    scores = {}
    for item in not_rated_items:
        # For each item, calculate weighted rating from all users
        ratings = user_item[item]
        weighted_scores = sim_scores * ratings
        scores[item] = weighted_scores.sum() / sim_scores.sum()

    # Sort scores
    recommended_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return recommended_items[:top_n]

def main():
    st.title("Simple User-based Collaborative Filtering Recommender")

    df = load_data()
    user_item = build_user_item_matrix(df)
    sim_df = compute_user_similarity(user_item)

    st.write(f"Dataset contains {df['user_id'].nunique()} users and {df['item_id'].nunique()} items.")

    user_input = st.number_input("Enter User ID (1 to 943):", min_value=1, max_value=943, value=1)

    if st.button("Get Recommendations"):
        recs = recommend_for_user(user_input, user_item, sim_df)
        if recs is None:
            st.error("User ID not found in dataset.")
        else:
            st.subheader(f"Top {len(recs)} recommendations for User {user_input}:")
            for i, (item_id, score) in enumerate(recs, start=1):
                st.write(f"{i}. Item ID: {item_id} (Score: {score:.2f})")

if __name__ == "__main__":
    main()
