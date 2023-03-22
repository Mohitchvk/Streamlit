from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error 

header = st.container()
dataset = st.container()
eda = st.container()
model_training = st.container()

@st.cache_data
def get_data():
    california_housing = fetch_california_housing(as_frame=True)
    return california_housing.frame
@st.cache_data
def holdout_split():
    X=get_data().drop(['AveBedrms','MedHouseVal','Longitude'],axis=1)
    y=get_data()['MedHouseVal']
    return train_test_split(X,y,test_size=0.3,random_state=0)

@st.cache_resource
def best_lamda_calc(x_train, x_test, y_train, y_test):
    rmse_list=[]
    for l in lambdas:
        model = Lasso(alpha=l, max_iter=10000).fit(x_train, y_train)
        y_pred = model.predict(x_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_list.append(rmse)
    return rmse_list

@st.cache_resource
def model_train(x_train, x_test, y_train, y_test, lambda_val=0.07):
    model = Lasso(alpha=lambda_val, max_iter=10000).fit(x_train, y_train)
    y_pred = model.predict(x_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    y_pred = model.predict(x_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return train_rmse, test_rmse





with header:
    st.title('California House Pricing Analysis🏙️!!...')
    st.write(" ")
    st.write("   ")
    st.write("   ")
    st.write("   ")
    st.write(" ")

with dataset:
    st.header('California Housing Dataset')
    st.text('This dataset is available from the Scikit Learn')
    st.write(get_data().head())
    st.write(" ")
    st.write("   ")
    st.write("   ")
    st.write("   ")
    st.write(" ")

with eda:
    st.header('Here we will explore the Data from the Dataset and make it tune the data to make it Linear model friendly')
    sel_col, disp_col = st.columns(2)
    sel_col.write(" ")
    sel_col.write("   ")
    sel_col.subheader('Check For the Distribution of a predictors')
    inp_feature = sel_col.selectbox('Select a predictor', options = list(get_data().columns))
    fig, ax = plt.subplots()
    get_data()[inp_feature].hist()
    disp_col.pyplot(fig)

    sel_col.write(" ")
    sel_col.write("   ")
    sel_col.write("   ")
    sel_col.write("   ")
    sel_col.write(" ")

    sel_col.subheader('Correlation of a predictors')
    corr_range = sel_col.slider("Correlation coefficient", min_value=0.0,max_value=1.0, step=0.1)
    corr = get_data().drop('MedHouseVal', axis=1).corr()
    corr=corr[((corr >= corr_range) | (corr <= -corr_range)) & (corr !=1.000)]
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    fig, ax = plt.subplots(figsize=(15, 15))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title("Correlation Matrix between Indepedent Features")
    disp_col.pyplot(fig)

    sel_col.write(" ")
    sel_col.write("   ")
    sel_col.write("   ")
    sel_col.write("   ")
    sel_col.write(" ")

    st.subheader("Correlation Between Predictors and Variables")
    f, ax = plt.subplots(figsize=(25, 1))
    corr = get_data().corr()
    treatment = corr.sort_values(by=['MedHouseVal'], ascending=False).head(1).T
    treatment = treatment.sort_values(by=['MedHouseVal'],ascending=False).T
    sns.heatmap(round(treatment,2), cmap=cmap, annot=True)
    plt.title("Correlation Between Independent and Target Variables")
    st.pyplot(f)

    
    sel_col.write(" ")
    sel_col.write("   ")
    sel_col.write("   ")
    sel_col.write("   ")
    sel_col.write(" ")

    sel_col.subheader("Distribution of the data points given predictors")
    X=sel_col.selectbox('Select 1st predictor', options = list(get_data().columns))
    Y=sel_col.selectbox('Select 2nd predictor', options = list(get_data().columns))

    fig, ax = plt.subplots(figsize=(15, 15))
    plt.scatter(get_data()[X],get_data()[Y])
    plt.title("Scatter plot between the Variables")
    disp_col.pyplot(fig)

    st.markdown("We need to avoid the multicoleniarity between the predictors inorder for the Linear Model to work well.")
    st.markdown("Here we can observe that the predictors AveBedrms and AveRooms are highly correlated and AveBedrms doesn't have much correlation with the target.")
    st.markdown("So we are dropping the predictor AveBedrms.")
    
    X=get_data().drop(['AveBedrms','MedHouseVal'],axis=1)
    y=get_data()['MedHouseVal']

    st.markdown('Predictors')
    st.write(X.head())
    st.markdown('Target')
    st.write(y.head())

    st.markdown("Also, we do train test split with 70:30 to keep a holdout set of unseen data")

    x_train, x_test, y_train, y_test = holdout_split()
    
    st.write(" ")
    st.write("   ")
    st.write("   ")
    st.write("   ")
    st.write(" ")


with model_training:
    st.header("Now Let's train the model and by playing with model parameters to find a model that best fits the given dataset")
    lambdas = np.arange(0,1.01,0.01)
    rmse_list = best_lamda_calc(x_train, x_test, y_train, y_test)
    
    fig, ax = plt.subplots(figsize=(15, 15))
    plt.plot(np.log(lambdas), rmse_list)
    plt.title("Change in lambda value VS rmse")
    st.pyplot(fig)
    st.markdown("We get the lowest rmse for the lambda value of:")
    min_lambda=lambdas[np.argmin(rmse_list)]
    st.write(min_lambda)


    st.markdown("Now lets train the model with lambda value of 0.07")
    lambda_val = st.slider("Set Lambda Val", min_value=0.00,max_value=1.01, step=0.01, value=0.07)

    train_rmse, test_rmse = model_train(x_train, x_test, y_train, y_test, lambda_val)
    st.markdown("The Train RMSE is:")
    st.write(train_rmse)
    st.markdown("The Test RMSE is:")
    st.write(test_rmse)

    




    