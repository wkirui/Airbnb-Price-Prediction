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
             This app uses data from the **Inside Airbnb Project**
             """)

# define function to load data
@st.cache
def load_data():
    listings_data = pd.read_csv("data/listings.csv.gz",
                                 compression='gzip',header = 0,sep=',',quotechar='"',error_bad_lines=False)
    return listings_data

# check top rows of the data
listings_data = load_data()
st.write("Here is the overview of the the airbnb listings")
st.write(listings_data.head())
st.write([x for x in listings_data.columns])

# berlin center: 52.521948, 13.413698

# def load_dataset():
#     listings_data1 = pd.read_csv("data/listings.csv.gz",
#                                  compression='gzip',header = 0,sep=',',quotechar='"',error_bad_lines=False)
#     st.write(listings_data1.head())
#     listings_data = pd.read_csv("data/listings.csv")
#     st.write("Listings Overview")
#     st.write(listings_data.head())
    
    
#     # listings_info = pd.DataFrame(listings_data.info())
    
#     st.write("Listings Data Summary")
#     st.write(listings_data.describe())
    
#     # let's add a sidebar for location
#     neighbourhood_group = listings_data['neighbourhood_group'].unique()
#     add_select_box = st.sidebar.selectbox(
#         'Slect the neighbourhood name',
#         (neighbourhood_group)
#     )
    
#     # create  a map that updates based on selected
#     # neighbourhood
#     st.text("Listings Distribution")
#     neighbourhood_selection = st.selectbox(
#         "Select Neighbourhood",
#         (neighbourhood_group))
#     # filter data based on selection
#     selected_neighbourhood_data = listings_data[listings_data['neighbourhood_group']==neighbourhood_selection]
    
#     # plot plot data for the selected neighbourhood
#     st.map(selected_neighbourhood_data)
#     # plot the distribution
    
    # return listings_data
# # load data
# st.write("Welcome to Berlin Airbnb data Analysis!")

if __name__ == "__main__":
    main()
