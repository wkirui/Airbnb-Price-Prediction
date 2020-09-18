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
import altair as alt
from math import radians,cos,sin,asin,sqrt

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
    st.write(""" 
             Looking at the price values, we can see that they currency denomination in dollars.\
                 
             We're going to clean them up so that we can start visualizing their distribution
             """)
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
    
    prices_dist = pd.DataFrame(listings_data['price'].describe())
    prices_dist = prices_dist.reset_index()
    prices_dist.columns = ['statistic','value']
    # st.write(len(listings_data[listings_data['price']==0]))
    st.write("""
             The following graph shows price distribution after cleaning it up.
             """)
    line_chart = alt.Chart(price_dist_sum).mark_line(interpolate='basis').encode(
    alt.X('price', title='Price ($)'),
    alt.Y('total', title='Total Listings'),
    ).properties(
        width = 700,height= 400,
        title='Price Distribution')
    st.altair_chart(line_chart)
    st.write("""
             We can make the following observations from the price distribution:
             - The average cost of renting an apartment is $74
             - Prices range from $0 to $9000
             - There are some apartments that do not have their prices indicated
             - 75% of the listings cost $75 or less
             
             As shown in this summary
             
             """)
    st.write(prices_dist)
    
    st.write("""
             Let's look at how some of the features such as distance from the city center, number of bedrooms and apartment type affect prices
             #### a) Distance from the city center
             - Generally prices are lower as you move away from the city center
             """)
    # calculate distance from city center
    listings_data_clean = calculate_distance_from_city_center(listings_data_clean,'latitude','longitude')
    
    # calculate price summary
    price_distance_summary = listings_data_clean.groupby('distance')['price'].mean().reset_index()
    
    # plot price distribution based on distance from city center
    distance_chart = alt.Chart(price_distance_summary).mark_line(interpolate='basis').encode(
    alt.X('distance', title='Distance (km)'),
    alt.Y('price', title='Price ($)'),
    ).properties(
        width = 700,height= 400,
        title='Average Prices by Distance from the City Center')
    st.altair_chart(distance_chart)
    
    # bedrooms
    st.write("""
            #### b) Number of bedrooms
             - Apartments with more bedrooms are more expensive
             """)
       # calculate price summary
    price_bedrooms_summary = listings_data_clean.groupby('bedrooms')['price'].mean().reset_index()
    
    # plot price distribution based on number of bedrooms
    num_bedrooms_chart = alt.Chart(price_bedrooms_summary).mark_line(interpolate='basis').encode(
    alt.X('bedrooms', title='No. of Bedrooms)'),
    alt.Y('price', title='Price ($)'),
    ).properties(
        width = 700,height= 400,
        title='Average Prices by Number of Bedrooms')
    st.altair_chart(num_bedrooms_chart)
    
    # room type
    st.write("""
            #### c) Room Type
            - Private rooms are cheaper
            - Hotel rooms cost up to 10 times more than other type of apartments
             """)
       # calculate price summary
    price_room_type_summary = listings_data_clean.groupby('room_type')['price'].mean().reset_index()
    price_room_type_summary = price_room_type_summary.sort_values(by='price',ascending=True)
    # plot price distribution based on room type
    room_type_chart = alt.Chart(price_room_type_summary).mark_bar().encode(
    alt.X('room_type', title='Room Type',sort=alt.EncodingSortField(field="price", order='ascending')),
    alt.Y('price', title='Price ($)'),
    order = 'price'
    ).properties(
        width = 700,height= 400,
        title='Average Prices by Room Type')
    st.altair_chart(room_type_chart)
    
    # Review Score
    st.write("""
             #### d) Review Score
              - Apartments with high review scores are cheaper on average
             """)
       # calculate price summary
    price_review_summary = listings_data_clean.groupby('review_scores_rating')['price'].mean().reset_index()
    
    # plot price distribution based on number of bedrooms
    review_score_chart = alt.Chart(price_review_summary).mark_line(interpolate='basis').encode(
    alt.X('review_scores_rating', title='Review Rating (%)'),
    alt.Y('price', title='Price ($)'),
    ).properties(
        width = 700,height= 400,
        title='Average Prices by Review Rating')
    st.altair_chart(review_score_chart)
    
        # Neighbourhood
    st.write("""
        #### d) Neighbourhood
         - Charlettenburg-Wilmersdof borough has the most expensive airbnb apartments on average ($114)
         - Reinickendorf apartments are the cheapest ($48)
        """)
       # calculate price summary
    price_neighbourhood_summary = listings_data_clean.groupby('neighbourhood_group_cleansed')['price'].mean().reset_index()
    price_neighbourhood_summary = price_neighbourhood_summary.sort_values(by='price',ascending=True)
    # plot price distribution based on number of bedrooms
    neighbourhood_chart = alt.Chart(price_neighbourhood_summary).mark_bar(interpolate='basis').encode(
    alt.Y('neighbourhood_group_cleansed', title='Neighbourhood',sort=alt.EncodingSortField(field="price", order='descending')),
    alt.X('price', title='Price ($)'),
    ).properties(
        width = 700,height= 400,
        title='Average Prices by Neighbourhood')
    st.altair_chart(neighbourhood_chart)
    
# # define function to load data
# @st.cache
def load_data():
    listings_data = pd.read_csv("data/listings.csv.gz",
                                 compression='gzip',header = 0,sep=',',quotechar='"',error_bad_lines=False,
                                 low_memory=False)
    return listings_data

# define function to calculate distance
# from the center of Berlin

# define harvesine function
def calculate_distance_from_city_center(df,lat,lon):
    """
    Calculates the great distance circle between
    two gps coordinates
    """
    # define berlin center coordinates
    center_lat = 52.521948
    center_lon = 13.413698
    
    # convert decimal degrees to radians
    df['lat1'] = df[lat].apply(lambda x : radians(x))
    df['lon1'] = df[lon].apply(lambda x : radians(x))
    lat2 = radians(center_lat)
    lon2 = radians(center_lon)
    
    # calculate harvesine distance
    df['dlat'] = lat2-df['lat1']
    df['dlon'] = lon2-df['lon1']
    
    df['sin_dlat'] = df['dlat'].apply(lambda x : sin(x/2)**2)
    df['cos_lat1'] = df['lat1'].apply(lambda x : cos(x))
    df['sin_dlon'] = df['dlon'].apply(lambda x : sin(x/2)**2)
    
    df['a'] = df['sin_dlat']+df['cos_lat1']*cos(lat2)*df['sin_dlon']
    df['distance'] = df['a'].apply(lambda x : round(2*asin(sqrt(x))*6371,3))
    
    # drop calculation columns
    df = df.drop(['sin_dlat','sin_dlon','cos_lat1','lat1','lon1','dlat','dlon','a'],axis=1)
    
    return df


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
