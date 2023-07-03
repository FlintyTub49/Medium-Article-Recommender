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
import os
from collections import Counter

PAGE_CONFIG = {"page_title" : "Medium Article Recommender", "layout" : "centered"}
st.set_page_config(layout = 'wide')

package_dir = os.path.dirname(os.path.abspath('app-v2.py'))
data_file = os.path.join(package_dir,'Data Files/data.csv.hdf5')
final = vaex.open(data_file)


# ----------------------------------
# Make The User Score Radar Graph
# ----------------------------------
def makeSingleGraph(score, topics):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r = score, theta = topics, fill = 'toself', name = 'User Score'))
    fig.update_layout(polar = dict(radialaxis = dict(visible = True, range = [0, 1]), bgcolor='#0E1117'), showlegend = True)
    return fig


# ----------------------------------
# Make The Nested Radar Plot
# ----------------------------------
def makeComparison(score, actual, topics):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r = score, theta = topics, fill = 'toself', name = 'User Score'))
    fig.add_trace(go.Scatterpolar(r = actual, theta = topics, fill = 'toself', name = 'Actual Score'))
    fig.update_layout(polar = dict(radialaxis = dict(visible = True, range = [0, 1]), bgcolor='#0E1117'), showlegend = True)
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
        return df[df['Search'].str.contains(text.strip())]


# ----------------------------------
# Plotting Distribution Of Topics
# ----------------------------------
def plotDistTopics(df):
    val = df['Max Prob Topic'].value_counts()
    val = pd.DataFrame(val).reset_index().rename(columns = {'index':'Topic', 0:'Count'})
    fig = px.histogram(val, y = 'Topic', x = 'Count', orientation = 'h')
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    return fig


# ----------------------------------
# Plotting Distribution Of Words
# ----------------------------------
def plotDistWords(df):
    filter = df[df['Actual Text'].str.count(' ') < 4500]
    count = (filter['Actual Text'].str.count(' ') + 1).tolist()
    fig = px.histogram(y = count, labels = {'y' : 'No. Of Words'}, orientation = 'h', marginal = 'box')
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    return fig


# --------------------------------------
# Plotting Distribution Of Reading Time
# --------------------------------------
def readingDist(df):
    demo = df[df['Reading Time'] < 30]
    fig = px.histogram(x = demo['Reading Time'].tolist(), labels = {'x': 'Reading Time'}, marginal = 'box')
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    return fig


# ----------------------------------
# Plotting Distribution Of Words
# ----------------------------------
@st.cache()
def topicBubble():
    cluster_file = os.path.join(package_dir, 'Data Files/clusters.csv')
    cluster = pd.read_csv(cluster_file)
    fig = px.scatter(cluster, x = 'X', y = 'Y', size = 'Count', color = 'Category', size_max = 50)
    fig.update_layout({
        'plot_bgcolor': '#0E1117',
        'paper_bgcolor': '#0E1117',
    })
    return fig

    
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
    # st.title('Medium Article Recommendation System')
    topic_names = ['Business and Startups', 'Fundamental Coding', 'Data Science & Statistics', 'Literature',
                   'Web Development', 'Social Media & Branding', 'Marketing & Sales', 'Team Dynamics',
                   'Cloud Development', 'Machine Learning & Deep Learning']
    
    # ----------------------------------
    # Side Bar Menu & Content
    # ----------------------------------
    st.sidebar.image(os.path.join(package_dir,'Data Files/medium.png'))
    menu = ['Recommendation System', 'Search Articles', 'Visualizations']
    choice = st.sidebar.selectbox('Menu', menu)

    
    # ----------------------------------
    # Main Recommendation System 
    # ----------------------------------
    if choice == 'Recommendation System':
        # ----------------------------------
        # Explainantion About The Page in Sidebar 
        # ----------------------------------
        st.sidebar.header('Recommendation System')
        st.sidebar.markdown('<p style="text-align:justify">Many times we find ourselves in positions where we want to read \
            an article based on a certain topic or topics but we do not know which is the best one for us.The app will help \
                us here.<br><br> The app will take your scores for each of the defined 10 topics and search our database for \
                    Medium articles which are best suited to you based on your preference scores.</p>', unsafe_allow_html=True)
        
        st.title('Medium Article Recommendation System')
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
        col1, col2 = st.beta_columns((1.25, 1.25))
        
        
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
        
        col2.header('Scores For Each Topic')
        col2.plotly_chart(bubble, use_container_width = True)
        col2.markdown(page_bg_img, unsafe_allow_html = True)
        
        
        # --------------------------------------
        # Printing The Distribution of Articels 
        # --------------------------------------
        col1.header('Spread Of Topics')
        fig = topicBubble()
        col1.plotly_chart(fig, width = 900)


        # --------------------------------------
        # Printing The Recommended Article 
        # --------------------------------------
        st.subheader('The Recommended Article is')
        st.write('')
        st.write('')
        st.title('{}'.format(title))
        st.subheader('Reading Time:   {}  minutes'.format(rt))
        st.write('')
        if a:
            image = os.path.join(package_dir,'Data Files/medium.png') 
            # st.markdown('[[Test]({})]({})'.format(image, link), unsafe_allow_html = True)
            st.subheader('[Go To Article]({})'.format(link))
        else:
            st.subheader('{}'.format(link))
        col2.markdown(page_bg_img, unsafe_allow_html = True)
    

    # ----------------------------------
    # All the Visualizations 
    # ----------------------------------
    elif choice == 'Visualizations':
        # ----------------------------------
        # Explainantion About The Page in Sidebar 
        # ----------------------------------
        st.sidebar.header('Visualizations')
        st.sidebar.markdown('<p style="text-align:justify">This section is about visualizations. Here we can get insights\
             about the data via various graphs that are generated on the basis of a particular topic or all topics all togther. \
                <br><br>The available visualizations are <ul style="text-align:justify"><li>Word Count of Articles</li><li>Reading \
                    Time Distribution of Articles</li><li>WordCloud for Articles</li><li>Word Similarities in \
                        Articles</li></p>', unsafe_allow_html=True)

        st.title('Visualizations')
        st.subheader('Select Appropriate Filters To Change Dashboard')
        st.write('')
        st.markdown(page_bg_img, unsafe_allow_html = True)


        # ----------------------------------
        # Query To Find In What Topic
        # ----------------------------------
        topics = ['All'] + topic_names
        topic = st.selectbox('Topics', topics)
        st.markdown(page_bg_img, unsafe_allow_html = True)


        # ----------------------------------
        # Show Which Graphs To Display
        # ----------------------------------
        f1, f2, f3, f4 = st.beta_columns((1.5, 1.5, 1.5, 1.2))
        if topic == 'All': text = 'Count Of Topics'
        else: text = 'WordCount of Articles'
        dist = f1.checkbox(text)
        wc = f2.checkbox('WordCloud')
        readT = f3.checkbox('Reading Time Distribution')
        wordSim = f4.checkbox('Word Similarities')
        
        
        _, col, _ = st.beta_columns((1, 2, 1))
        col.write('')

        if not dist and not wc and not readT and not wordSim:
            col.subheader('Select Visualizations To Display')

        else:
            if dist:
                # --------------------------------------
                # Showing Distribution Plot Or WordPlot
                # --------------------------------------
                if topic == 'All':
                    col.header('Distribution of All The Topics')
                    distTopic = plotDistTopics(final)
                    col.plotly_chart(distTopic, height = 1200)

                else:
                    col.header('Distribution of No. of Words in {} Articles'.format(topic))
                    filter = filtTopic(df = final, topic = topic)
                    distWord = plotDistWords(filter)
                    col.plotly_chart(distWord, height = 300)


            if wc:
                # ----------------------------------
                # Showing The Appropriate WordCloud
                # ----------------------------------
                col.header('WordCloud For {} Articles'.format(topic))
                wc = os.path.join(package_dir,'wcClouds/{}.jpg'.format(topic))
                col.image(wc, caption = 'WordCloud For {} Articles'.format(topic))


            if readT:
                # ----------------------------------
                # Showing The Reading Time Split
                # ----------------------------------
                col.header('Reading Time For Articles in {}'.format(topic))
                filter = filtTopic(df = final, topic = topic)
                readTime = readingDist(filter)
                col.plotly_chart(readTime)

            
            if wordSim:
                # ----------------------------------
                # Showing The Appropriate WordCloud
                # ----------------------------------
                col.header('Word Similarities For {} Topics'.format(topic))
                sim20 = os.path.join(package_dir,'wordRelations/{}.jpg'.format(topic))
                col.image(sim20, caption = 'Word Similarities For {}'.format(topic))


    # ----------------------------------
    # Find Article Based On Keywords 
    # ----------------------------------
    elif choice == 'Search Articles':
        # ----------------------------------
        # Explainantion About The Page in Sidebar 
        # ----------------------------------
        st.sidebar.header('Visualizations')
        st.sidebar.markdown('<p style="text-align:justify">If we do not want to get an recommended article and just look for \
            articles on our own based on certain conditons, we can do that here.<br>The kind of filter available on this \
                page are:<ul style="text-align:justify"><li><u>Keyword:</u> Search for an Article Title based on a particular \
                    keyword</li><li><u>Topic:</u> You can search for articles in all topics or just in some specific topic</li>\
                        <li><u>Reading Time:</u> If you want an article that can be finished in a particular amount of time, you can \
                            specify that here</li></ul></p>', unsafe_allow_html=True)


        st.title('Search Articles')
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
            # Filtering Keyword
            display = filtKeyword(df = final, text = text)
            
            # Filtering Time
            display = filtTime(df = display, time1 = time1, time2 = time2)
            
            # Filtering Topic
            display = filtTopic(df = display, topic = topic)
            
            if display.shape[0] > 0:
                # Checking if we have 50 values to print, otherwise we print the entire subset
                if display.shape[0] >= 15: 
                    samp = 15
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
