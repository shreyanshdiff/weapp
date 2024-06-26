from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import ydata_profiling
import pandas as pd
from streamlit_ydata_profiling import st_profile_report
import os 

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.title("ML GO BRRRR!!!!!")
    choice = st.radio("Navigation", ["Upload","Profiling Details ","Modelling", "Download"])
    st.info("This project helps you to build and explore data in an efficient manner ")

if choice == "Upload":
    st.title("Upload Your Dataset ")
    file = st.file_uploader("Upload Your Dataset here")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling": 
    st.title("EDA")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Modelling": 
    chosen_target = st.selectbox('Choose your Target Column', df.columns)
    if st.button('Run'): 
        setup(df, target=chosen_target, remove_outliers=True)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")