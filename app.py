import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

st.title('Bank Marketing Outcome Prediction')
age = st.slider('Age', 18, 70)
default = st.selectbox('Has credit in default?', ['No', 'Yes'])
balance = st.number_input('Average Yearly Balance (in Euros)', min_value=-9000, max_value=30000, value=0)
housing = st.selectbox('Has housing loan?', ['No', 'Yes'])
loan = st.selectbox('Has personal loan?', ['No', 'Yes'])
day = st.slider('Last contact day of the month', 1, 31)
duration = st.number_input('Last contact duration (in seconds)', min_value=0, max_value=2000)
campaign = st.slider('Number of contacts performed during this campaign', 0, 25)
pdays = st.number_input('Number of days passed by after last contacted from a previous campaign (-1 if client was not previously contacted)',
                                 min_value=-1, max_value=400)
marital = st.selectbox('Marital Status', ['Married', 'Single', 'Divorced'])
previous = st.number_input('Number of contacts performed before this campaign', min_value=0, max_value=15)
month = st.selectbox('Month (last contact )', ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 
       'Dec'])
job = st.selectbox('Job', ['Management', 'Technician', 'Entrepreneur', 'Blue-Collar', 
                                    'Retired', 'Admin', 'Services', 'self-Employed', 
                                    'Unemployed', 'Housemaid', 'Student'])

education = st.selectbox('Education', ['Tertiary', 'Secondary', 'Primary'])



# contact = left_column.selectbox('Contact communicaiton type', ['telephone', 'unknown', 'cellular'])
# poutcome = left_column.selectbox('Outcome of the previous marketing campaign', ['other', 'unknown', 'success', 'failure'])





button = st.button('Predict')

X = {'age': [age],
        'default': [0],
        'balance': [balance],
        'housing': [0],
        'loan': [0],
        'day': [day],
        'duration': [duration],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'divorced': [0],
        'married': [0],
        'single': [0],
        'Apr': [0],
        'Aug': [0],
        'Dec': [0],
        'Feb': [0],
        'Jan': [0],
        'Jul': [0],
        'Jun': [0],
        'Mar': [0],
        'May': [0],
        'Nov': [0],
        'Oct': [0],
        'Sep': [0],
        'Admin': [0],
        'Blue-Collar': [0],
        'Entrepreneur': [0],
        'Housemaid': [0],
        'Management': [0],
        'Retired': [0],
        'Self-employed': [0],
        'Service': [0],
        'Student': [0],
        'Technician': [0],
        'Unemployed': [0],
        'Primary': [0],
        'Secondary': [0],
        'Tertiary': [0]}

if default == 'Yes':
        X.update({'default': [1]})

if housing == 'Yes':
       X.update({'housing': [1]})

if loan == 'Yes':
       X.update({'loan': [1]})

if marital == 'Divorced':
        X.update({'divorced': [1]})
elif marital == 'Married':
       X.update({'married': [1]})
else:
       X.update({'single': [1]})

def category(dict, key):
        if key in dict:
            dict.update({key: [1]})

category(X, month)
category(X, job)
category(X, education)

df = pd.DataFrame.from_dict(X).values
scaler = MinMaxScaler()
data = scaler.fit_transform(df)
model = pickle.load(open('rfc_model.sav', 'rb'))

if button:
    pred_proba = model.predict_proba(df)
    prediction = (pred_proba[:,-1] > 0.7).astype(int)
    if prediction == 1:
           st.write(':bold[Client would subscribe]')
    else:
           st.write(':bold[Client would not subscribe]')

