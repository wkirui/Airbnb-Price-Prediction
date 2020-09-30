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
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error,mean_absolute_error
import pickle
import joblib
from sklearn.model_selection import RandomizedSearchCV

# instantiate app
def main():
    st.write("""
             ## Airbnb Price Prediction
             
             In this analysis, I look at airbnb apartment prices in Berlin, Germany. 
             Airbnb data for most cities and towns around the world is made publicly 
             available through [**Inside Airbnb**](http://insideairbnb.com/) project.
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
        try:
            listings_data_clean = clean_prices_column(listings_data,col)
        except KeyError:
            pass
    
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
    st.write("Here is the quick summary of the prices distribution")
    st.write(prices_dist)
    st.write("""
             - The average cost of renting an apartment is $68 per day
             - Prices range from $0 to 8000
             - 75% of the listings cost $50 or less
             - 5% of the listings cost more than the 95th percentile ($155)
             - 0.3% (70) of the listings cost more than $500
             
             """)
    
    # # prices greater than 95th percentile
    # median_price = listings_data_clean['price'].quantile(0.5)
    # pct_95 = listings_data_clean['price'].quantile(0.95)
    # outlier_df = listings_data_clean[listings_data_clean['price']>pct_95]
    # more_than_500_df = listings_data_clean[listings_data_clean['price']>500]
    
    # st.write("Median Price",median_price,
    #          "95th Percentile:",pct_95,
    #          "Above 95th Percentile:",len(outlier_df),round(len(outlier_df)/len(listings_data_clean)*100,3),
    #          "Above $500:",len(more_than_500_df),round(len(more_than_500_df)/len(listings_data_clean)*100,3)
    #          )
    
    # st.write(more_than_500_df.head(10))
    
    # let's see how this changes if we replace outliers with median prices
    # st.write("""
    #          For this Analysis, we will cap the prices at $500.\
    #         In order to do so, we replace prices that are more than $500 with the median price which is $50
    #         """)
    # listings_data_clean = listings_data_clean.copy()
    # listings_data_clean = listings_data_clean[(listings_data_clean['price']>0) & (listings_data_clean['price']<=500)]
    # # listings_data_clean['price'] = np.where(listings_data_clean['price']>500,50,listings_data_clean['price'])
    # prices_dist_trim = pd.DataFrame(listings_data_clean['price'].describe())
    # prices_dist_trim = prices_dist_trim.reset_index()
    # prices_dist_trim.columns = ['statistic','value']
    # st.write(prices_dist_trim,
    #          "- Note that the standard deviation went down from 229 to 63 which means we now have less variability in the data")
    
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
    # limit max prices to $500
    price_distance_summary = listings_data_clean[listings_data_clean['price']<500].groupby('distance')['price'].median().reset_index()
    price_distance_summary['price'] = np.round(price_distance_summary['price'],0)
    price_distance_summary['distance'] = np.round(price_distance_summary['distance'],1)
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
             - Apartments with more bedrooms cost more than those with fewer bedrooms on average
             """)
       # calculate price summary
    price_bedrooms_summary = listings_data_clean.groupby('bedrooms')['price'].mean().reset_index()
    # price_bedrooms_summary['price'] = price_bedrooms_summary['price'].astype(int) 
    
    # plot price distribution based on number of bedrooms
    num_bedrooms_chart = alt.Chart(price_bedrooms_summary).mark_line(interpolate='basis').encode(
    alt.X('bedrooms', title='No. of Bedrooms)'),
    alt.Y('price', title='Price ($)'),
    ).properties(
        width = 700,height= 400,
        title='Average Prices by Number of Bedrooms')
    st.altair_chart(num_bedrooms_chart)
    # st.write(price_bedrooms_summary['bedrooms'].unique())
    
    # room type
    st.write("""
            #### c) Room Type
            - Shared rooms are cheaper on average
            - Hotel rooms cost up to 3 times more than private rooms
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
              - There is no clear relationship between prices and review scores
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
    # st.write(price_review_summary.head())
        # Neighbourhood
    st.write("""
        #### d) Neighbourhood
         - Charlettenburg-Wilmersdof borough has the most expensive apartments ($81) followed by Mitte ($77) and Pankow ($75) boroughs
         - Reinickendorf apartments are the cheapest ($50)
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
        
    # generate total listings by neighbourhood
    # st.write(listings_data_clean['neighbourhood_group_cleansed'].value_counts().reset_index())
    
    # Map Visualization
    # st.text("Listings Distribution")
    st.write("""
             ### Map Visualization
             Let's look at appartment listings distribution by borough on a map
             """)
    neighbourhood_group = [x for x in price_neighbourhood_summary['neighbourhood_group_cleansed'].unique()]
    neighbourhood_selection = st.selectbox(
        "Select Neighbourhood",
        (sorted(neighbourhood_group)))
    # filter data based on selection
    selected_neighbourhood_data = listings_data_clean[listings_data_clean['neighbourhood_group_cleansed']==neighbourhood_selection][:500]
    
    # plot plot data for the selected neighbourhood
    st.map(selected_neighbourhood_data)
    
    # feature selection
    st.write("""
             ### Feature Selection
             #### a) Calculate missing values
             In this section we select the best features to work with in our model
              - First we calculate the percentage of missing values in each column
              - We will drop features with more than 50% missing values. There are only 5 columns
              
             """)
    # listings_data_clean = drop_columns_with_missing_vals(listings_data_clean)
    
    # columns with missing values
    # columns_with_missing_values = listings_data_clean.isnull().sum().rename('total').reset_index().sort_values(by='total',ascending=False)
    # columns_with_missing_values.columns =['feature','total']
    # columns_with_missing_values['pct_missing_values'] = round(columns_with_missing_values['total']/len(listings_data_clean)*100,1)
    # columns_to_drop = columns_with_missing_values[columns_with_missing_values['pct_missing_values']>50]['feature']
    # st.write(len(columns_to_drop))
    # st.write(columns_to_drop)
    
    # drop columns with more than 50% missing values
    # listings_data_clean = drop_columns_with_missing_vals(listings_data_clean)
    
    # clean up percent columns
    pct_cols = ['host_acceptance_rate','host_response_rate']
    for col in pct_cols:
        listings_data_clean[col] = listings_data_clean[col].str.replace('%','')
        listings_data_clean[col] = pd.to_numeric(listings_data_clean[col],errors='coerce')
        
    # st.write(listings_data_clean[['host_acceptance_rate','host_response_rate']].head(10))
    # st.write(listings_data_clean[['host_acceptance_rate','host_response_rate']].isnull().sum())
    
    # drop columns with more than 50% missing values
    listings_data_clean = drop_columns_with_missing_vals(listings_data_clean)
    
        # drop some specific columns
    listings_data_clean = listings_data_clean.drop(['scrape_id','host_id','last_scraped','calendar_last_scraped'],axis=1)
    
    # clean up host acceptance rate
    # listings_data_clean['host_acceptance_rate'] = listings_data_clean['host_acceptance_rate'].astype(int,errors='ignore')
    # st.write(listings_data_clean['host_acceptance_rate'].unique())
    #check data info
    # separate columns with integer & object values
    columns_with_int_or_float_vals = [x for x in listings_data_clean.columns if listings_data_clean[x].dtype==int
                                      or listings_data_clean[x].dtype==float]
    columns_with_categorical_vals = [x for x in listings_data_clean.columns if listings_data_clean[x].dtype==object]
    
    # st.write(columns_with_int_or_float_vals)
    # st.write(columns_with_categorical_vals)
    
    # drop categorical columns with highest dimension
    # set max at 50 unique values
    high_dimension_columns = []
    for i in columns_with_categorical_vals:
        if listings_data_clean[i].nunique()>200:
            high_dimension_columns.append(i)
    # st.write(high_dimension_columns)
    # drop high dimension columns
    listings_data_clean = listings_data_clean.drop(high_dimension_columns,axis=1)
    
    # update categorical columns list
    columns_with_categorical_vals = [x for x in listings_data_clean.columns if listings_data_clean[x].dtype==object]
    
    
    # impute columns
    st.write(
        """
        #### b) Impute missing values
          - Calculate median for each of the integer columns and use it to fill the missing values
          - Use 'unknown' to fill missing values in columns with categorical values
        """
    )
    # Impute missing values
    # int/float columns
    for col in columns_with_int_or_float_vals:
        if col != 'id':
            median_val = np.round(np.median(listings_data_clean[listings_data_clean[col].isnull()==False][col]),0)
            listings_data_clean[col] = listings_data_clean[col].fillna(median_val)
    # categorical columns
    for col in columns_with_categorical_vals:
        listings_data_clean[col] = listings_data_clean[col].fillna('unknown')
     
    # Encode categorical columns   
    st.write("""
             #### c) Encode categorical columns
            - We use One-Hot Encoding method to encode categorical columns. Specifically we use pandas' **get_dummies** method
            - The resulting DataFrame has 87 columns
            """)
    # encode values
    encoded_listings_data = pd.get_dummies(listings_data_clean,
                                           columns=columns_with_categorical_vals,
                                           drop_first=True)
    st.write(encoded_listings_data.shape,
             encoded_listings_data.head())
    
    # drop apartments without prices
    encoded_listings_data = encoded_listings_data[encoded_listings_data['price']>0]
    
    # let's drop some columns to avoid collinearity
    drop_cols = ['host_listings_count','latitude','longitude',
                 'host_total_listings_count','minimum_nights_avg_ntm',
                 'maximum_nights_avg_ntm']
    encoded_listings_data = encoded_listings_data.drop(drop_cols,axis=1)
    
    # Feature importance
    st.write(""" 
             ### Feature Importance
             After cleaning up the data, the next step is to select the most important features for our model.
             These are the features that have high influence on the target variable (price).
             
             We train a RandomForest Regression model on our features then we select the best features by iteratively training our model on the top **n** features from the important features list.
             Our final model will use features that produced the best score.
             
             """)
    # check if we have a saved csv file
    # if error, generate feature importances
    try:
        top_features_selected_df = pd.read_csv("trained_models/feature_importances.csv")
        st.write("Top 10 Features",
            top_features_selected_df.head(10))
    except FileNotFoundError:
        # generate top 10 features
        with st.spinner('Please Hang on ...'):
            top_features_selected_df = generate_important_features(encoded_listings_data)
            st.write(top_features_selected_df.head(10))  
        # notify done
        st.success('Done!')

    # create our model
    st.write("""
             ### Modeling
             We will train our model using RandomForest Regression. Price is a continuous variable and hence we need a regression algorithm
             to make our prediction.\n
             If we're not satisfied with the results of this algorithm, we can try other algorithms such as XGBoost for performance comparison.
             
             """)
    # prepare data for modeling
    # replace outliers with median values
    # encoded_listings_data['price'] = np.where(encoded_listings_data['price']>encoded_listings_data['price'].quantile(0.95),
    #                                           np.median(encoded_listings_data['price']),
    #                                           encoded_listings_data['price'])
    
    # try different set of features
    # features_scores = {}
    # for i in range(1,len(top_features_selected_df['feature'])+1):
        
    #     top_n_features_list = [x for x in top_features_selected_df['feature'][:i]]
    #     X = encoded_listings_data[top_n_features_list]
    #     y = encoded_listings_data['price']

    #     # split data into train and test
    #     X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
    #     best_features_model = RandomForestRegressor(verbose=1,n_jobs=-1,random_state=42)
    #     best_features_model.fit(X_train,y_train)
    #     # check performance
    #     y_pred = best_features_model.predict(X_test)
    #     mse = mean_squared_error(y_test,y_pred)
    #     rmse = mse**(1/2)
    #     features_scores[i] = rmse
    
    # create features scores df
    # features_score_df = pd.DataFrame(features_scores.items(),columns=['number_of_features','rmse'])
    # features_score_df = features_score_df.sort_values(by='rmse',ascending=False)
    # st.write(features_score_df)
    
    n = 185
    top_n_features_list = [x for x in top_features_selected_df['feature'][:n]]
    X = encoded_listings_data[top_n_features_list]
    y = encoded_listings_data['price']

    # split data into train and test
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42)
        
    # load saved model or create one
    # create model name
    trained_model_name = "trained_models/rf_trained_model_v1.pkl"
    try:
        # add progress bar
        with st.spinner("Loading Model..."):
            # load file
            with open(trained_model_name,'rb') as f:
                model = joblib.load(f)
        # st.success('Woop!Woop!')
    except FileNotFoundError:
    # create a model
        model = RandomForestRegressor(verbose=0,n_jobs=-1,random_state=42)
        model.fit(X_train,y_train)
        with open(trained_model_name,'wb') as f:
            joblib.dump(model,f,compress=3)
    
    # check model
    # st.write(model.get_params())
    
    # make predictions with the model
    with st.spinner("Making Predictions..."):
        y_pred = model.predict(X_test)
    # scores = cross_val_score(model,X_test,y_test,scoring='r2')
    # st.write(scores)

    # get R^2 score
    model_score = model.score(X_test,y_test)
    mse = mean_squared_error(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    st.write("Model Results:")
    st.write("R-Squared:",round(model_score*100,2))
    st.write("MSE:",round(mse,4))
    st.write("RMSE:",round(mse**(1/2),4))
    st.write("MAE:",round(mae,4))
    st.write( """
             The model's $R^2$ score is 14%. This means our model is performing poorly at explaining the variability in our dataset.\
                 
             The root mean squared error is also  97.14 which means that we have a $97 error between our predicted prices and the actual prices! This is quite high and can possibly be explained by the big outliers in our prices.
             
             The MAE of 23.7 is however lower and we can use it to explain the performance of our model. Mean absolute error is not affected by outliers in the data.
             
             Below are the top 10 predictions from our model
             """)
    
    # Let's look at top predictions from our model
    predicted_df = pd.DataFrame(y_pred,columns=['predicted'])
    y_test = y_test.reset_index(drop=True)
    actual_v_predictions_df = pd.concat([y_test,predicted_df],axis=1,sort=False)
    actual_v_predictions_df['predicted'] = np.round(actual_v_predictions_df['predicted'],1)
    # calculate errors
    actual_v_predictions_df['error'] = actual_v_predictions_df['predicted'] - actual_v_predictions_df['price']
    st.write(actual_v_predictions_df.head(10))
    
    # hyperparameter tuning
     # define parameters to optimize
    n_estimators = [int(x) for x in np.linspace(start=200,stop=2000,num=10)]
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
    # load saved model
    hyperparam_model = "trained_models/rf_hypermodel_v1.pkl"
    # load saved mode or
    # train one
    try:
        # create progress bar to show model loading progress
        with st.spinner('Loading Hyperparameter model...'):
            # load file
            with open(hyperparam_model,'rb') as f:
                rf_random = joblib.load(f)
        # st.success('Wooooo!')
    except FileNotFoundError:
        # search best parameters
        rf_model = RandomForestRegressor()
        # search using 3 fold cross validation
        rf_random = RandomizedSearchCV(estimator=rf_model,param_distributions=random_grid,
                                       n_iter=50,cv=2,verbose=0,random_state=42,n_jobs=-1)
        # fit the random search
        rf_random.fit(X_train,y_train)
        # open saved model
        with open(hyperparam_model,'wb') as f:
            joblib.dump(rf_random,f,compress=3)
    
    # hyperparameter tuning
    st.write("""
             #### Hyperparameter Tuning
             Our model did not perform optimally as expected. Let's use RandomSearchCV to find the best parameters that we can use to improve the performance of our model.
             
             These are the parameters to optimize for our model:
             - number of estimators
             - max depth
             - min samples split
             - min samples leaf
             - bootstrap
             
             We will use the following algorithm to find the best parameters for our prediction
             """)
    # print random search model
    st.write(rf_random)
    
    # get best parameters
    st.write("Running the algorithm above we get the following results")
    st.write(rf_random.best_params_)
    
    st.write("Let's retrain our model using the parameters above and compare the results with our first model")
    
    # define hyperparameters
    n_estimators = rf_random.best_params_['n_estimators']
    max_depth = rf_random.best_params_['max_depth']
    min_samples_split = rf_random.best_params_['min_samples_split']
    min_samples_leaf = rf_random.best_params_['min_samples_leaf']
    bootstrap = rf_random.best_params_['bootstrap']
    
    # train the model with hyperparameters
    hyper_model_name = "trained_models/rf_hypermodel_final_v1.pkl"
    try:
        with open(hyper_model_name,'rb') as f:
            hyper_model = joblib.load(f)
    except FileNotFoundError:
        
    # create a model using hyperparameters
        hyper_model = RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            bootstrap=bootstrap,verbose=0,
                                            n_jobs=-1,random_state=42)
        hyper_model.fit(X_train,y_train)
        with open(hyper_model_name,'wb') as f:
            joblib.dump(hyper_model,f,compress=3)
    
    st.write("Here is our model with the hyperparameters",
             
             hyper_model)
    
    # make predictions with the model
    with st.spinner("Loading hyperparameter model predictions..."):
        y_pred = hyper_model.predict(X_test)
    # get R^2 score
    model_score = hyper_model.score(X_test,y_test)
    mse = mean_squared_error(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    st.write("Model Results:")
    st.write("R-Squared:",round(model_score*100,2))
    st.write("MSE:",round(mse,4))
    st.write("RMSE:",round(mse**(1/2),4))
    st.write("MAE:",round(mae,4))
    st.write("""
             - Our RMSE went up from 97 to 104
             - $R^2$ also decreased from 14% to 0.9%  and the MAE increased from 23 to 30. 
             
             Our final model is not performing as expected and hence we need to try a few more tweaks in order to improve performance. 
             For instance we can experiment with different sets of features or train a different algorithm.
             
             Here are the top predictions from our final model. Generally some predictions are quite ok but we need to further minimize our error rate.
             """)
    
    predicted_df = pd.DataFrame(y_pred,columns=['predicted'])
    y_test = y_test.reset_index(drop=True)
    actual_v_predictions_df = pd.concat([y_test,predicted_df],axis=1,sort=False)
    actual_v_predictions_df['predicted'] = np.round(actual_v_predictions_df['predicted'],1)
    actual_v_predictions_df['error'] = actual_v_predictions_df['predicted'] - actual_v_predictions_df['price']
    
    # show predictions
    st.write(actual_v_predictions_df.head(10))
    
    # check model score
    # scores = cross_val_score(hyper_model,X_test,y_test,scoring='neg_mean_absolute_error',cv=3,n_jobs=-1,verbose=1)
    # st.write(scores)
    
    
    
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
        
    drops columns with 50% missing values
    """
    percent_missing_values = round((df.isnull().sum()/len(df))*100,2)
    percent_missing_values = pd.DataFrame({"column_name":df.columns,
                                        "%_missing_values":percent_missing_values})
    percent_missing_values = percent_missing_values.reset_index(drop=True)

    # sort by % of missing values
    percent_missing_values = percent_missing_values.sort_values(by='%_missing_values',ascending=False)
    # print(len(percent_missing_values[percent_missing_values['%_missing_values']>30]))
    columns_missing_more_than_50_pct_vals = [x for x in percent_missing_values[percent_missing_values['%_missing_values']>50]['column_name']]
    df = df.drop(columns_missing_more_than_50_pct_vals,axis=1)
    
    return df

# define feature importance calculation function
@st.cache
def generate_important_features(df):
    """
    df: dataframe
    return: feature importances
    """
    # Create model using randomregressor

    # prepare data for modeling
    X = df.drop(['id','price'],axis=1)
    y = df['price']
    
    # split the data
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    
    # load or create a model
    saved_features_model = "trained_models/rf_features_model_v1.sav"
    try:
        # check if we have saved the model
        with open(saved_features_model,'rb') as f:
            model = pickle.load(f)
    except:
        # model the data
        model = RandomForestRegressor(n_estimators=500,n_jobs=-1)
        model.fit(X_train,y_train)
        # save the model
        with open(saved_features_model,'wb') as f:
            pickle.dump(model,f)

    # get feature importance
    feat_importances = model.feature_importances_
    col_list = [x for x in X_train.columns]
    feat_list = {}
    for i,v in enumerate(feat_importances):
    #     print("Feature: %0d, Score: %.5f" %(i,v))
        feat_list[col_list[i]]= v

    # create feature importance dataframe
    feature_importances_df = pd.DataFrame(feat_list.items(),columns=['feature','score'])
    feature_importances_df = feature_importances_df.sort_values('score',ascending=False)
    
    # save features
    feature_importances_df.to_csv('trained_models/feature_importances.csv',index=False)

    # return top features
    return feature_importances_df


if __name__ == "__main__":
    main()
