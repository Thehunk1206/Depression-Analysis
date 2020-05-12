import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import io

s = io.StringIO()

def getUserInfo(filename):
    csv_sep = filename.split(".")
    user_info = csv_sep[0].split(" ")
    #user_info is a list containing name and age at pos 1 and 2
    return user_info[1],user_info[2]
    

def main():
    st.title("Depression Analyzer")
    st.header("Visualize emotion data from CSV file")

    module1_file = st.file_uploader("upload Module1 file",type=['csv'])
    module2_file = st.file_uploader("upload Module2 file",type=['csv'])

    if module1_file  is not None:
        s.write(module1_file)
        s.seek(0)
        st.write(s.read())
        '''
        name,age = getUserInfo(module1_file)
        st.write(name,age)
        '''
        
main()

