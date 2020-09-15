# In this project I perform EDA on Berlin airbnb data
# and create a machine learning model to predict rent prices
# in a given location

# The project is inspired by this kaggle notebook
# https://www.kaggle.com/brittabettendorf/predicting-prices-xgboost-feature-engineering

# import module
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# instantiate app
def main():
    st.write("""
             ## Welcome to Berlin Airbnb Data Analysis
             This app uses data from the [**Inside Airbnb Project**](http://insideairbnb.com/)
             """)
     # check top 10 rows of the data
    listings_data = load_data()
    st.write("Here is an overview of the airbnb listings data")
    st.write(listings_data.head(10))
    
    # clean up price columns
    st.write(" Looking at the price values, we can see that they currency denomination in dollars.\
             We're going to clean them up so that we can start visualizing their distribution")
    # define a function to clean up price values
    def clean_prices_column(df,col):
        df[col] = df[col].str.replace('$','')
        df[col] = df[col].str.split('.',expand=True)[0]
        df[col] = df[col].str.replace(',','')
        df[col] = df[col].astype(int,errors='ignore')
        return df
    # create list of columns to clean up
    price_cols = ['price','weekly_price','monthly_price','extra_people',
                  'security_deposit','cleaning_fee']
    for col in price_cols:
        listings_data_clean = clean_prices_column(listings_data,col)
    
    # plot price distribution
    price_dist_sum = listings_data.groupby('price')['id'].count().rename('total').reset_index().sort_values(by='price',ascending=True)
    st.write(price_dist_sum['price'].describe())
    
    plt.scatter('price','total',data=price_dist_sum)
    plt.title("Price Distribution")
    plt.xlabel("Price ($)")
    plt.ylabel("Total Listings")
    st.pyplot()

# define function to load data
# @st.cache
def load_data():
    listings_data = pd.read_csv("data/listings.csv.gz",
                                 compression='gzip',header = 0,sep=',',quotechar='"',error_bad_lines=False,
                                 low_memory=False)
    return listings_data


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
