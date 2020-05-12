import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff


import io

s = io.StringIO()

def getUserInfo(filename):
    csv_sep = filename.split(".")
    user_info = csv_sep[0].split(" ")
    #user_info is a list containing name and age at pos 1 and 2
    return user_info[1],user_info[2]

@st.cache 
def load_csv(file):
    return pd.read_csv(file)

def main():
    st.title("Depression Analyzer")
    st.header("Visualize emotion data from CSV file")

    module1_file = st.file_uploader("Upload Module1 file",type=['csv'])
    module2_file = st.file_uploader("Upload Module2 file",type=['csv'])

    if module1_file  is not None:
        df = load_csv(module1_file)

        #======For pie chart====================
        emotion_labels = df["emotions"].unique()
        values = []
        emotion_count = []
        for l in emotion_labels:
            islabel = df["emotions"] == l
            #st.write(df[islabel]["emotions"])
            lab_count = df[islabel]["emotions"].count()
            values.append(lab_count)
            
        
        hist_fig = px.histogram(df["emotions"], x="emotions",
                   marginal="violin")

        pie_fig = go.Figure(data=[go.Pie(labels=emotion_labels, values=values, hole=.3)])
        
        st.markdown("## Module 1")
        st.subheader("Visualizing facial emotion data")

        
        if st.button("Produce charts"):
            
            st.markdown("* Pie Chart")
            st.plotly_chart(pie_fig)

            st.markdown("* Histogram")
            st.plotly_chart(hist_fig)

main()

