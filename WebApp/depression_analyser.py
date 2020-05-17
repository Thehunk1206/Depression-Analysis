import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff


@st.cache 
def load_csv(file):
    return pd.read_csv(file)

def main():
    st.title("Depression Analyzer")
    st.header("Visualize emotion data from CSV file")

    module1_file = st.file_uploader("Upload Module1 file(csv file only)",type=['csv'])
    module2_file = st.file_uploader("Upload Module2 file(csv file only)",type=['csv'])

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
    if module2_file is not None:
        df2 = load_csv(module2_file)
        st.markdown("## Module 2")
        st.subheader("Visualizing response sentiment data")
        
        st.markdown("* Responses and Sentiment")
        st.write(df2[["text","sentiment"]])
        
        sentiment_label = ["NEGATIVE","POSITIVE"]
        sentiment_values = [df2[df2["sentiment"]=="NEGATIVE"]["sentiment"].count(),df2[df2["sentiment"]=="POSITIVE"]["sentiment"].count()]
        pie_fig_sentiment = go.Figure(data=[go.Pie(labels=sentiment_label, values=sentiment_values, hole=.3)])

        st.markdown("* Pie Chart")
        st.plotly_chart(pie_fig_sentiment)
        
main()
