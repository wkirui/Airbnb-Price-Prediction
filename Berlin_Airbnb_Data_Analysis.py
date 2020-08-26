# In this project I perform EDA on Berlin airbnb data
# and create a machine learning model to predict rent prices
# in a given location

# The project is inspired by this kaggle notebook
# https://www.kaggle.com/brittabettendorf/predicting-prices-xgboost-feature-engineering

# import module
import pandas as pd
import numpy as np
import streamlit as st

# instantiate app
def main():
    st.write("""
             # Welcome to Berlin Airbnb Data Analysis
             This app uses data from the *Inside Airbnb Project*
             """)
    load_dataset()

def load_dataset():
    listings_data = pd.read_csv("data/listings.csv")
    st.write("Listings Overview")
    st.write(listings_data.head())
    
    # listings_info = pd.DataFrame(listings_data.info())
    
    st.write("Listings Data Summary")
    st.write(listings_data.describe())
    
    # let's add a sidebar for location
    neighbourhood_group = listings_data['neighbourhood_group'].unique()
    add_select_box = st.sidebar.selectbox(
        'Slect the neighbourhood name',
        (neighbourhood_group)
    )
    st.text("Listings Distribution")
    st.map(listings_data)
    # plot the distribution
    
    # return listings_data
# # load data
# st.write("Welcome to Berlin Airbnb data Analysis!")

if __name__ == "__main__":
    main()
