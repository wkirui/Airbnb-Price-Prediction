# In this project I perform EDA on Berlin airbnb data
# and create a machine learning model to predict rent prices
# in a given location

# The project is inspired by this kaggle notebook
# https://www.kaggle.com/brittabettendorf/predicting-prices-xgboost-feature-engineering

# import module
import pandas as pd
import numpy as np
from datetime import datetime,date
import time
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from math import radians,cos,sin,asin,sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.model_selection import RandomizedSearchCV

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
    price_dist_sum = listings_data_clean.groupby('price')['id'].count().rename('total').reset_index().sort_values(by='price',ascending=True)
    
    prices_dist = pd.DataFrame(listings_data_clean['price'].describe())
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
             - 5% of the listings cost more than the 95th percentile ($155)
             - 0.5% (129) of the listings cost more than $500
             
             As shown in this summary
             
             """)
    st.write(prices_dist)
    # # prices greater than 95th percentile
    # median_price = listings_data_clean['price'].quantile(0.5)
    # outlier_list = listings_data_clean['price'].quantile(0.95)
    # outlier_df = listings_data_clean[listings_data_clean['price']>400]
    # st.write("Median Price",median_price,"95th Percentile:",outlier_list,"Above 95th Percentile:",len(outlier_df))
    # st.write(outlier_df.head(15))
    
    # let's see how this changes if we replace outliers with median prices
    st.write("""
             For this Analysis, we will cap the prices at $500.\
            In order to do so, we replace prices that are more than $500 with the median price which is $50
            """)
    listings_data_clean = listings_data_clean.copy()
    listings_data_clean = listings_data_clean[(listings_data_clean['price']>0) & (listings_data_clean['price']<=500)]
    # listings_data_clean['price'] = np.where(listings_data_clean['price']>500,50,listings_data_clean['price'])
    prices_dist_trim = pd.DataFrame(listings_data_clean['price'].describe())
    prices_dist_trim = prices_dist_trim.reset_index()
    prices_dist_trim.columns = ['statistic','value']
    st.write(prices_dist_trim,
             "- Note that the standard deviation went down from 229 to 63 which means we now have less variability in the data")
    
    # # calculate log of price
    # st.write("### Log of Prices")
    # listings_data_clean_x = listings_data_clean[listings_data_clean['price']>0]
    # price_log_vals = pd.DataFrame(np.log(listings_data_clean_x['price']))
    # price_log_sum = price_log_vals.describe()
    # st.write(price_log_sum)
    # sns.distplot(price_log_vals)
    # # st.pyplot()
    # st.line_chart(price_log_vals)
    
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
            - Hotel rooms cost up to 3 times more than other type of apartments
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
         - Mitte borough has the most expensive apartments on average ($72) followed by Charlettenburg-Wilmersdof borough ($70)
         - Reinickendorf apartments are the cheapest ($45)
        """)
       # calculate price summary
    price_neighbourhood_summary = listings_data_clean.groupby('neighbourhood_group_cleansed')['price'].mean().reset_index()
    price_neighbourhood_summary = price_neighbourhood_summary.sort_values(by='price',ascending=True)
    # st.write(price_neighbourhood_summary)
    # plot price distribution based on number of bedrooms
    neighbourhood_chart = alt.Chart(price_neighbourhood_summary).mark_bar(interpolate='basis').encode(
    alt.Y('neighbourhood_group_cleansed', title='Neighbourhood',sort=alt.EncodingSortField(field="price", order='descending')),
    alt.X('price', title='Price ($)'),
    ).properties(
        width = 700,height= 400,
        title='Average Prices by Neighbourhood')
    st.altair_chart(neighbourhood_chart)
    
    # visualize distribution on the map
    # st.text("Listings Distribution")
    st.write("""
             ### Map Visualization
             We can use a map to visualize how these listings are distributed
             """)
    neighbourhood_group = [x for x in price_neighbourhood_summary['neighbourhood_group_cleansed'].unique()]
    neighbourhood_selection = st.selectbox(
        "Select Neighbourhood",
        (sorted(neighbourhood_group)))
    # filter data based on selection
    selected_neighbourhood_data = listings_data_clean[listings_data_clean['neighbourhood_group_cleansed']==neighbourhood_selection]
    
    # plot plot data for the selected neighbourhood
    st.map(selected_neighbourhood_data)
    
    # feature selection
    st.write("""
             ### Feature Selection
             In this section we select the best features to work with in our model
              - First we calculate the percentage of missing values in each column
              - Then we drop features with mode than 30% missing values
              - We have 21 columns in this category which include: 
                  
                  - 'interaction', 'host_about', 
                  'neighborhood_overview', 'transit','security_deposit', 'host_acceptance_rate',
                  'space', 'cleaning_fee'
             """)
    # drop columns with more than 30% missing values
    listings_data_clean = drop_columns_with_missing_vals(listings_data_clean)
    
    # filter apartments in Germany only
    listings_data_clean = listings_data_clean[listings_data_clean['country_code']=='DE']
    
    #  let's define columns to use
    # create list of potential features
    working_list = ['id','host_is_superhost',
                    'host_neighbourhood', 'host_listings_count', 'host_total_listings_count',
                    'host_identity_verified', 'neighbourhood_cleansed',
                    'neighbourhood_group_cleansed', 'city', 'state',
                    'latitude', 'longitude', 'is_location_exact', 'property_type',
                    'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'bed_type',
                    'price', 'guests_included', 'extra_people', 'minimum_nights', 'maximum_nights',
                    'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights', 
                    'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm',
                    'has_availability', 'availability_30', 'availability_60', 
                    'availability_90', 'availability_365', 'number_of_reviews',
                    'number_of_reviews_ltm','review_scores_rating', 
                    'review_scores_accuracy','review_scores_cleanliness', 'review_scores_checkin',
                    'review_scores_communication', 'review_scores_location', 'review_scores_value', 
                    'requires_license', 'instant_bookable', 'is_business_travel_ready', 'cancellation_policy',
                    'require_guest_profile_picture', 'require_guest_phone_verification',
                    'calculated_host_listings_count', 'calculated_host_listings_count_entire_homes', 
                    'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms',
                    'reviews_per_month', 'distance']
    # select only columns in our list
    listings_data_clean = listings_data_clean[working_list]
    st.write("- We then select the following 57 columns that potentially affect the airbnb prices",
        listings_data_clean.head())
    
    # encode categorical columns
    st.write("""
             - Next we encode columns with categorical columns using One-Hot Encoding method. Specifically using pandas' **get_dummies** method
             - The Resulting DataFrame has 537 columns! We definitely need to scale down the number of features.
             """)
    # define columns to encode
    columns_to_encode = ['host_neighbourhood','host_is_superhost','host_identity_verified','is_location_exact','neighbourhood_cleansed','neighbourhood_group_cleansed', 'city','state',
                    'property_type','room_type','bed_type','requires_license','has_availability',
                     'instant_bookable', 'is_business_travel_ready', 'cancellation_policy',
                     'require_guest_profile_picture', 'require_guest_phone_verification']
    # encode selected columns
    encoded_listings_data = pd.get_dummies(listings_data_clean,
                                       columns=columns_to_encode)
    st.write(encoded_listings_data.head(8))
    
    st.write("""
             - We also impute missing values for the integer columns their mean values. Basically we calculate the mean values for these columns then we use them to fill the NAs
             """)
    # Impute missing values for integer columns
    listings_int_values_data = encoded_listings_data.iloc[:,1:40]
    total_vals_missing = listings_int_values_data.isnull().sum()
    listings_int_values_df = pd.DataFrame({'column_name':listings_int_values_data.columns,
                            'total_missing_vals':total_vals_missing})
    listings_int_values_df = listings_int_values_df.reset_index(drop=True).sort_values('total_missing_vals',ascending=False)

    # define list of columns to impute
    columns_to_impute = [x for x in listings_int_values_df[listings_int_values_df['total_missing_vals']>0]['column_name']]

    # impute missing values with mean
    for col in columns_to_impute:
        mean_val = round(np.mean(encoded_listings_data[col]),1)
        encoded_listings_data[col] = encoded_listings_data[col].fillna(mean_val)
    
    st.write(""" 
             ### Feature Importance
             - After cleaning up the data, the next step is to select the most important features for our model. These are the features that have high influence on the target variable (price)
             """)
    
    # generate top 50 features
    with st.spinner('Please Hang on ...'):
        top_features_selected_df = generate_important_features(encoded_listings_data,50)
        st.write(top_features_selected_df)  
    # st.balloons()
    st.success('Done!')
    # 

    # create our model
    st.write("""
             ### Modeling
             With our selected features, we can now go ahead and train the model. 
             We will use RandomForestRegressor for our baseline model
             
             """)
    # prepare data fro modeling
    top_n_features_list = [x for x in top_features_selected_df['feature']]
    X = encoded_listings_data[top_n_features_list]
    y = encoded_listings_data['price']

    # split data into train and test
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    
    # load saved model or create one
    # create model name
    trained_model_name = "trained_models/rf_model_v1.sav"
    try:
        with open(trained_model_name,'rb') as f:
            model = pickle.load(f)
    except:
    # create a model
        model = RandomForestRegressor(max_depth=5,n_jobs=-1,random_state=42)
        model.fit(X_train,y_train)
        with open(trained_model_name,'wb') as f:
            pickle.dump(model,f)
    
    # check model
    st.write(model.get_params())
    # make predictions with the model
    y_pred = model.predict(X_test)

    # get R^2 score
    model_score = model.score(X_test,y_test)
    mse = mean_squared_error(y_test,y_pred)
    st.write("Model Results:")
    st.write("R-Squared:",round(model_score*100,2))
    st.write("MSE:",round(mse,4))
    st.write("RMSE:",round(mse**(1/2),4))
    st.write( """
             The model's R^2 score is 42% R^2. This means our model is not performing well at explaining the variability in our dataset.\
                 
             The root mean squared error is 37.7 which means e have a $37 difference between our predicted prices and the actual prices which we can further minimize in order to improve our model.
             """)
    
    # Let's look at top predictions from our model
    predicted_df = pd.DataFrame(y_pred,columns=['predicted'])
    y_test = y_test.reset_index(drop=True)
    actual_v_predictions_df = pd.concat([y_test,predicted_df],axis=1,sort=False)
    st.write("Here are the top prediction results",
        actual_v_predictions_df.head(10))
    
    # hyperparameter tuning
     # define parameters to optimize
    n_estimators = [int(x) for x in np.linspace(start=50,stop=500,num=20)]
    max_features = ['auto','sqrt']
    max_depth = [int(x) for x in np.linspace(10,110,num=11)]
    max_depth.append(None)
    min_samples_split = [2,5,10]
    min_samples_leaf = [1,2,4]
    bootstrap = [True,False]
     
     # define search grid
    random_grid = {
         'n_estimators': n_estimators,
         'max_features': max_features,
         'max_depth': max_depth,
         'min_samples_split': min_samples_split,
         'min_samples_leaf': min_samples_leaf,
         'bootstrap': bootstrap
        }
    
    # search best parameters
    rf_model = RandomForestRegressor()
    # search using 3 fold cross validation
    rf_random = RandomizedSearchCV(estimator=rf_model,param_distributions=random_grid,
                                   n_iter=100,cv=3,verbose=2,random_state=42,n_jobs=-1)
    # fit the random search
    rf_random.fit(X_train,y_train)
    
    # get best parameters
    st.write(rf_random.best_params_)


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

# define function to drop columns
def drop_columns_with_missing_vals(df):
    """
    @df: dataframe\
        
    drops columns with 30% missing values
    """
    percent_missing_values = round((df.isnull().sum()/len(df))*100,2)
    percent_missing_values = pd.DataFrame({"column_name":df.columns,
                                        "%_missing_values":percent_missing_values})
    percent_missing_values = percent_missing_values.reset_index(drop=True)

    # sort by % of missing values
    percent_missing_values = percent_missing_values.sort_values(by='%_missing_values',ascending=False)
    # print(len(percent_missing_values[percent_missing_values['%_missing_values']>30]))
    columns_missing_more_than_30_pct_vals = [x for x in percent_missing_values[percent_missing_values['%_missing_values']>30]['column_name']]
    df = df.drop(columns_missing_more_than_30_pct_vals,axis=1)
    
    return df

# define feature importance calculation function
@st.cache
def generate_important_features(df,n):
    """
    df: dataframe
    n: number of features to select
    """
    # Create model using randomregressor

    # prepare data for modeling
    X = df.drop(['id','price'],axis=1)
    y = df['price']
    
    # load or create a model
    saved_features_model = "trained_models/rf_features_model_v1.sav"
    try:
        # check if we have saved the model
        with open(saved_features_model,'rb') as f:
            model = pickle.load(f)
    except:
        # model the data
        model = RandomForestRegressor()
        model.fit(X,y)
        # save the model
        with open(saved_features_model,'wb') as f:
            pickle.dump(model,f)

    # get feature importance
    feat_importances = model.feature_importances_
    col_list = [x for x in X.columns]
    feat_list = {}
    for i,v in enumerate(feat_importances):
    #     print("Feature: %0d, Score: %.5f" %(i,v))
        feat_list[col_list[i]]= v

    # create feature importance dataframe
    feature_importances_df = pd.DataFrame(feat_list.items(),columns=['feature','score'])
    feature_importances_df = feature_importances_df.sort_values('score',ascending=False)

    # create a list of top 50 features
    top_n_features_list = [x for x in feature_importances_df[:n]['feature']]

    top_n_features_df = feature_importances_df[0:n]

    return top_n_features_df


if __name__ == "__main__":
    main()
