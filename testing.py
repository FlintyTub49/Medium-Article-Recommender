
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import streamlit.components.v1 as components
import random
import webbrowser
import vaex

PAGE_CONFIG = {"page_title" : "Medium Article Recommender", "layout" : "centered"}
st.set_page_config(layout = 'wide')
final = vaex.open('https://github.com/FlintyTub49/Medium-Article-Recommender/blob/main/Data%20Files/final.csv.hdf5')


# ----------------------------------
# Make The User Score Radar Graph
# ----------------------------------
def makeSingleGraph(score, topics):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r = score, theta = topics, fill = 'toself', name = 'User Score'))
    fig.update_layout(polar = dict(radialaxis = dict(visible = True, range = [0, 1])), showlegend = True)
    return fig


# ----------------------------------
# Make The Nested Radar Plot
# ----------------------------------
def makeComparison(score, actual, topics):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r = score, theta = topics, fill = 'toself', name = 'User Score'))
    fig.add_trace(go.Scatterpolar(r = actual, theta = topics, fill = 'toself', name = 'Actual Score'))
    fig.update_layout(polar = dict(radialaxis = dict(visible = True, range = [0, 1])), showlegend = True)
    return fig


# ----------------------------------
# Get Recommendations Based On User Score
# ----------------------------------
def getRec(topic_names, rad_values):
    topic_array = np.array(final[topic_names])
    sc = MinMaxScaler(feature_range = (0, 1))
    norm = sc.fit_transform(topic_array.T).T

    mini = 1000
    index = []
    for i in range(len(final)):
        y = mean_squared_error(norm[i], rad_values)
        if y < mini: 
            index.append(i)
            mini = y
    top3 = list(reversed(index[-3:]))
    return top3, [norm[top3[0]], norm[top3[1]], norm[top3[2]]]


# ----------------------------------
# Filtering Based on Topic
# ----------------------------------
def filtTopic(df, topic):
    if topic == 'All': 
        return df
    else:
        return df[df['Max Prob Topic'] == topic]


# ----------------------------------
# Filtering Based on Reading Time
# ----------------------------------
def filtTime(df, time1, time2):
    df = df[df['Reading Time'] >= time1]
    return df[df['Reading Time'] <= time2]
    
    
# ----------------------------------
# Filtering Based on Keyword
# ----------------------------------
def filtKeyword(df, text):
    if text == '' or text == ' ':
        return df
    else:
        return df[df['Titles'].str.contains(text)]

    
# ----------------------------------
# For Centering Stuff On Page
# ----------------------------------
page_bg_img = '''
        <style>
        body {
        color:white;
        text-align:center;
        }
        </style>
        '''


# ----------------------------------
# Main App Running
# ----------------------------------
def main():
    st.title('Medium Article Recommendation System')
    topic_names = ['Business and Startups', 'Fundamental Coding', 'Data Science & Statistics', 'Literature',
                   'Web Development', 'Social Media & Branding', 'Marketing & Sales', 'Team Dynamics',
                   'Cloud Development', 'Machine Learning & Deep Learning']
    
    # ----------------------------------
    # Side Bar Menu
    # ----------------------------------
    menu = ['Recommendation System', 'Search Articles']
    choice = st.sidebar.selectbox('Menu', menu)
    
    
    # ----------------------------------
    # Main Recommendation System 
    # ----------------------------------
    if choice == 'Recommendation System':
        st.subheader('A Recommendation Systemn for Medium Articles Based on Topic Selection')
        st.markdown(page_bg_img, unsafe_allow_html=True)

        
        # ----------------------------------
        # Sliders To Get User Score
        # ----------------------------------
        st.subheader('Tune The Sliders to Get Recommendations Accordingly')
        st.title('')

        _, col3, _, col4, _ = st.beta_columns((0.1, 1, 0.25, 1, 0.1))
        startups = col3.slider(label = 'Business and Startups', min_value = 0.0, max_value = 1.0, step = 0.05, 
                               value = 0.2, key = 'Business and Startups')
        coding = col3.slider(label = 'Fundamental Coding', min_value = 0.0, max_value = 1.0, step = 0.05, 
                             value = 0.2, key = 'Fundamental Coding')
        dsstats = col3.slider(label = 'Data Science & Statistics', min_value = 0.0, max_value = 1.0, step = 0.05, 
                              value = 0.2, key = 'Data Science & Statistics')
        lite = col3.slider(label = 'Literature', min_value = 0.0, max_value = 1.0, step = 0.05, 
                           value = 0.2, key = 'Literature')
        webdev = col3.slider(label = 'Web Development', min_value = 0.0, max_value = 1.0, step = 0.05, 
                             value = 0.2, key = 'Web Development')
        somedia = col4.slider(label = 'Social Media and Branding', min_value = 0.0, max_value = 1.0, step = 0.05, 
                              value = 0.2, key = 'Social Media and Branding')
        marksal = col4.slider(label = 'Marketing and Sales', min_value = 0.0, max_value = 1.0, step = 0.05, 
                              value = 0.2, key = 'Marketing And Sales')
        tedyna = col4.slider(label = 'Team Dynamics', min_value = 0.0, max_value = 1.0, step = 0.05, 
                             value = 0.2, key = 'Team Dynamics')
        cloud = col4.slider(label = 'Cloud Development', min_value = 0.0, max_value = 1.0, step = 0.05, 
                            value = 0.2, key = 'Cloud Development')
        mldl = col4.slider(label = 'Machine Learning & Deep Learning', min_value = 0.0, max_value = 1.0, step = 0.05, 
                           value = 0.2, key = 'Machine Learning & Deep Learning')
        rad_values = [startups, coding, dsstats, lite, webdev, somedia, marksal, tedyna, cloud, mldl]
        
        st.write('')
        a = st.button('Get Recommendation')
        col1, _, col2 = st.beta_columns((1.2, 0.15, 1.25))
        
        
        # ----------------------------------
        # Getting The Recommended Articles 
        # ----------------------------------
        if a:
            indices, actual = getRec(topic_names = topic_names, rad_values = rad_values)
            rand = random.randint(0, 2)
            title = final[indices[rand]][0]
            rt = final[indices[rand]][3]
            link = final[indices[rand]][4]
            actual = actual[rand]
        else:
            title = 'Your Recommended Article Title Will Show Up Here'
            rt = 'N/A'
            link = "Get An Article Recommended To Get It's URL"
        
        
        # ----------------------------------
        # Making & Presenting The Graph 
        # ----------------------------------
        if a == 0:
            bubble = makeSingleGraph(score = rad_values, topics = topic_names)
        else:
            bubble = makeComparison(score = rad_values, actual = actual, topics = topic_names)
        
        title
        col1.subheader('Scores For Each Topic')
        col1.plotly_chart(bubble, use_container_width = True)
        col1.markdown(page_bg_img, unsafe_allow_html = True)
        
        
        # ----------------------------------
        # Printing The Recommended Article 
        # ----------------------------------
        col2.subheader('The Recommended Article is')
        col2.write('')
        col2.write('')
        col2.title('{}'.format(title))
        col2.subheader('Reading Time:   {}  minutes'.format(rt))
        col2.write('')
        if_link = '<iframe src="' + link + '"><i/frame>'
        if a:
            col2.write(if_link, unsafe_allow_html = True)
            col2.write('')
            if col2.button('Go To Page'):
                webbrowser.open_new_tab(link)
        else:
            col2.subheader('{}'.format(link))
        col2.markdown(page_bg_img, unsafe_allow_html = True)
    
    
    # ----------------------------------
    # Find Article Based On Keywords 
    # ----------------------------------
    elif choice == 'Search Articles':
        st.header('Select Appropriate Filters To Find Article')
        st.subheader('')
        text = st.text_input('Enter Keyword To Search Through Entire Database')
        col1, _, col2 = st.beta_columns((1, 0.1, 1))
    
        # ----------------------------------
        # Query To Find In What Topic
        # ----------------------------------
        topics = ['All'] + topic_names
        topic = col1.selectbox('Topics', topics)
        
        
        # ----------------------------------
        # Query To Filter According To Reading Time
        # ----------------------------------
        time1, time2 = col2.slider(label = 'Reading Time (mins)', min_value = 1, max_value = 60, step = 1, 
                           value = (5, 10), key = 'Reading Time')
        
        
        # ----------------------------------
        # Query To Search And Display Output
        # ----------------------------------
        st.write('')
        _, mid, _ = st.beta_columns((1, 4, 1))
        mid.markdown(page_bg_img, unsafe_allow_html = True)
        search = mid.button('Search Database')
        st.write('')
        
        
        # ----------------------------------
        # Searching According To The Queries text, topic, time1, time2
        # ----------------------------------
        _, midi, _ = st.beta_columns((0.3, 5.4, 0.3))
        if search:
            # Filtering Topic
            display = filtTopic(df = final, topic = topic)
            
            # Filtering Time
            display = filtTime(df = display, time1 = time1, time2 = time2)
            
            # Filtering Keyword
            display = filtKeyword(df = display, text = text)
            
            if display.shape[0] > 50:
                # Checking if we have 50 values to print, otherwise we print the entire subset
                if display.shape[0] >= 40: 
                    samp = 50
                else:
                    samp = display.shape[0]
                
                display = display[['Titles', 'Max Prob Topic','Reading Time', 
                                   'Actual Text', 'Story URL']].sample(samp)
                display = display.to_pandas_df()
                midi.dataframe(display, width = 10000, height = 400)
            
            else:
                midi.subheader('No Data Present For This Filter Combination')
                
        else: midi.subheader('Enter Search To Generate Results')
        
if __name__ == '__main__':
    main()
