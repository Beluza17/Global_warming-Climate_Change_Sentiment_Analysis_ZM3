#streamlit dependencies
import streamlit as st
import joblib, os

## data dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nlppreprocess import NLP # pip install nlppreprocess
from nltk import pos_tag
import base64
import seaborn as sns
import io

from collections import Counter
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords, wordnet  
from sklearn.feature_extraction.text import CountVectorizer   
from sklearn.feature_extraction.text import TfidfTransformer 
from spacy.pipeline import TextCategorizer
from textblob import TextBlob

import warnings
warnings.filterwarnings('ignore')

from nlppreprocess import NLP
nlp = NLP()

import matplotlib.style as style 
sns.set(font_scale=1.5)
style.use('seaborn-pastel')
style.use('seaborn-poster')
from PIL import Image
from wordcloud import WordCloud

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')





df = pd.read_csv('train.csv')

def TweetCleaner(tweet):
    
    """This function uses regular expressions to remove url's, mentions, hashtags, 
       punctuation, numbers and any extra white space from tweets after converting 
       everything to lowercase letters.
    """
    tweet = re.sub(r'^RT ','', re.sub(r'https://t.co/\w+', '', tweet).strip())         # Remove url's
    tweet = tweet.lower()                                    # Convert everything to lowercase 
    tweet = re.sub('@[\w]*','',tweet)                        # Remove mentions  
    tweet = re.sub(r'#\w*', '', tweet)                       # Remove hashtags
    tweet = re.sub(r'\d+', '', tweet)                        # Remove numbers
    tweet = re.sub(r"[,.;':@#?!\&/$]+\ *", ' ', tweet)       # Remove punctuation
    tweet = re.sub(r"U+FFFD ", ' ', tweet)                   # Remove that funny diamond
    tweet = re.sub(r'\s\s+', ' ', tweet)                     # Remove extra whitespace
    tweet = tweet.lstrip(' ')                                # Remove space in front of tweet  
       
    emojis = re.compile("["
                           u"\U0001F600-\U0001F64F"  # removes emoticons,
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

    tweet = emojis.sub(r'', tweet)

    return tweet






# Clean the tweets in the message column
df['message'] = df['message'].apply(TweetCleaner)

def getAnalysis(sentiment):
    if sentiment == 0:
        return 'Neutral'
    elif sentiment == 1:
        return 'Pro'
    elif sentiment == -1:
        return 'Anti'
    else: 
        return 'News'
df['analysis'] = df['sentiment'].apply(getAnalysis)






#Lemmatization - reduce the word into dictionary root form
def lemma1(df):
    
    """
    This function modifies the original train dataframe.
    A new column for the length of each tweet is added.
    The tweets are then tokenized and each word is assigned a part of speech tag 
    before being lemmatized
    """
    
    tokenized = df.apply(word_tokenize)
    pos_tags = tokenized.apply(nltk.tag.pos_tag)

    def get_wordnet_pos(tag):

        if tag.startswith('J'):
            return wordnet.ADJ

        elif tag.startswith('V'):
            return wordnet.VERB

        elif tag.startswith('N'):
            return wordnet.NOUN

        elif tag.startswith('R'):
            return wordnet.ADV
    
        else:
            return wordnet.NOUN
        
    wnl = WordNetLemmatizer()
    df = get_wordnet_pos(pos_tags)
    df = wnl.lemmatize(df)
    
    return df




def lemma(df):
    
    """
    This function modifies the original train dataframe.
    A new column for the length of each tweet is added.
    The tweets are then tokenized and each word is assigned a part of speech tag 
    before being lemmatized
    """
    df['length'] = df['message'].str.len()
    df['tokenized'] = df['message'].apply(word_tokenize)
    df['pos_tags'] = df['tokenized'].apply(nltk.tag.pos_tag)

    def get_wordnet_pos(tag):

        if tag.startswith('J'):
            return wordnet.ADJ

        elif tag.startswith('V'):
            return wordnet.VERB

        elif tag.startswith('N'):
            return wordnet.NOUN

        elif tag.startswith('R'):
            return wordnet.ADV
    
        else:
            return wordnet.NOUN
        
    wnl = WordNetLemmatizer()
    df['pos_tags'] = df['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])
    df['lemmatized'] = df['pos_tags'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])
    df['lemmatized'] = [' '.join(map(str, l)) for l in df['lemmatized']]  
    return df

df = lemma(df)







#Create a function to get the subjectivity
def getSubjectivity(tweet):
    return TextBlob(tweet).sentiment.subjectivity

#create function to get polarity
def getPolarity(tweet):
    return TextBlob(tweet).sentiment.polarity

#create two new columns
df['subjectivity'] = df['message'].apply(getSubjectivity)
df['polarity'] = df['message'].apply(getPolarity)   






def frequency(tweet):
    
    """
    This function determines the frequency of each word in a collection of tweets 
    and stores the 25 most frequent words in a dataframe, 
    sorted from most to least frequent     
    """
    cv = CountVectorizer(stop_words='english')                             # Count vectorizer excluding english stopwords
    words = cv.fit_transform(tweet)
    
    # Count the words in the tweets and determine the frequency of each word
    sum_words = words.sum(axis=0)
    words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    
    # Create a dataframe to store the top 25 words and their frequencies
    frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])
    frequency = frequency.head(20)
    
    return frequency

# Extract the top 25 words in each class
pro_frequency = frequency(df['lemmatized'][df['sentiment']==1])
anti_frequency = frequency(df['lemmatized'][df['sentiment']==-1])
news_frequency = frequency(df['lemmatized'][df['sentiment']==2])
neutral_frequency = frequency(df['lemmatized'][df['sentiment']==0])






st.set_option('deprecation.showPyplotGlobalUse', False)

st.sidebar.title('Options')

option = st.sidebar.selectbox('Which Dashboard?', ('Home','Predictions', 'Charts', 'Raw Tweets','Contact Us'))

st.header(option)





if option == 'Home':
    st.subheader('Tweets Classifier App')
    from PIL import Image
    image = Image.open('resources/imgs/personalaction-kidsclimatechange-1200x630.jpg')
    st.image(image, caption='Which Tweet are you?', use_column_width=True)






vectorizer = open('resources/tfidfvect.pkl','rb')   ##  will be replaced by the cleaning and preprocessing function
tweet_cv = joblib.load(vectorizer)

if option == 'Predictions':
    st.subheader('Climate Change Tweet Classifier')


    st.info('Make Predictions of your Tweet(s) using our ML Model')

    data_source = ['Select option', 'Single text'] ## differentiating between a single text and a dataset inpit

    source_selection = st.selectbox('What to classify?', data_source)

    # Load Our Models
    def load_prediction_models(model_file):
        #predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
        loaded_models = joblib.load(open(os.path.join(model_file),"rb"))
        return loaded_models


    # Getting the predictions
    def get_keys(val,my_dict):
        for key,value in my_dict.items():
            if val == value:
                return key


    if source_selection == 'Single text':
        ### SINGLE TWEET CLASSIFICATION ###
        st.subheader('Single tweet classification')

        input_text = st.text_area('Enter Text (max. 140 characters):') ##user entering a single text to classify and predict
        all_ml_models = ["lr","nb","rf","knn", "EDSA", "lsvc_op"]
        model_choice = st.selectbox("Choose ML Model",all_ml_models)

        #st.info('for more information on the above ML Models please visit: https://datakeen.co/en/8-machine-learning-algorithms-explained-in-human-language/')

        prediction_labels = {'Negative':-1,'Neutral':0,'Positive':1,'News':2}


        if st.button('Classify'):

            st.text("Original test :\n{}".format(input_text))
            text1 = TweetCleaner(input_text) ###passing the text through the 'cleaner' function
            data = io.StringIO(text1)
            text1 = pd.read_csv(data)
            #text1 = lemma1(text1)
            vect_text = tweet_cv.transform([text1]).toarray()

            if model_choice == 'lr':
                predictor = load_prediction_models("lr.pkl")
                prediction = predictor.predict(vect_text)
                # st.write(prediction)
            elif model_choice == 'rf':
                predictor = load_prediction_models("rf.pkl")
                prediction = predictor.predict(vect_text)
                # st.write(prediction)
            elif model_choice == 'nb':
                predictor = load_prediction_models("nb.pkl")
                prediction = predictor.predict(vect_text)
                # st.write(prediction)
            elif model_choice == 'knn':
                predictor = load_prediction_models("knn.pkl")
                prediction = predictor.predict(vect_text)
				# st.write(prediction)
            elif model_choice == 'EDSA':
                predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
                prediction = predictor.predict(vect_text)
            
            elif model_choice == 'lsvc_op':
                predictor = joblib.load(open(os.path.join("resources/lsvc_op.pkl"),"rb"))
                prediction = predictor.predict(vect_text)

             
            final_result = get_keys(prediction,prediction_labels)
            st.success("Tweets Categorized as: {}".format(final_result))
                
                
            #csv = input_text.to_csv(index=False)
               
            #b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
            #href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'

            #st.markdown(href, unsafe_allow_html=True)






if option == 'Charts':
    st.subheader('Lets Visualize')
    
    #Show the value counts

    df['analysis'].value_counts()

    #plot and visualize the counts
    plt.title('Sentiment Analysis')
    plt.xlabel('Sentiment')
    plt.ylabel('Counts')
    df['analysis'].value_counts().plot(kind='bar')
    st.pyplot()

    style.use('seaborn-pastel')

    fig, axes = plt.subplots(ncols=2, 
                            nrows=1, 
                            figsize=(20, 10), 
                            dpi=100)

    sns.countplot(df['sentiment'], ax=axes[0])

    labels=['Pro', 'News', 'Neutral', 'Anti'] 

    axes[1].pie(df['sentiment'].value_counts(),
                labels=labels,
                autopct='%1.0f%%',
                shadow=True,
                startangle=90,
                explode = (0.1, 0.1, 0.1, 0.1))

    fig.suptitle('Tweet distribution', fontsize=20)
    st.pyplot()
    


    # Plot the distribution of the length tweets for each class using a box plot
    sns.boxplot(x=df['sentiment'], y=df['length'], data=df, palette=("Blues_d"))
    plt.title('Tweet length for each class')
    st.pyplot()

    # Extract the words in the tweets for the pro and anti climate change classes 
    anti_words = ' '.join([twts for twts in anti_frequency['word']])
    pro_words = ' '.join([twts for twts in pro_frequency['word']])
    news_words = ' '.join([twts for twts in news_frequency['word']])
    neutral_words = ' '.join([twts for twts in neutral_frequency['word']])

    # Create wordcloud for the anti climate change class
    anti_wordcloud = WordCloud(width=800, 
                            height=500, 
                            random_state=110, 
                            max_font_size=110, 
                            background_color='white',
                            colormap="Reds").generate(anti_words)

    # Create wordcolud for the pro climate change class
    pro_wordcloud = WordCloud(width=800, 
                          height=500, 
                          random_state=73, 
                          max_font_size=110, 
                          background_color='white',
                          colormap="Greens").generate(pro_words)

    # Create wordcolud for the news climate change class
    news_wordcloud = WordCloud(width=800, 
                            height=500, 
                            random_state=0, 
                            max_font_size=110, 
                            background_color='white',
                            colormap="Blues").generate(news_words)

    # Create wordcolud for the neutral climate change class
    neutral_wordcloud = WordCloud(width=800, 
                            height=500, 
                            random_state=10, 
                            max_font_size=110, 
                            background_color='white',
                            colormap="Oranges").generate(neutral_words)



    # Plot pro and anti wordclouds next to one another for comparisson
    f, axarr = plt.subplots(2,2, figsize=(35,25))
    axarr[0,0].imshow(pro_wordcloud, interpolation="bilinear")
    axarr[0,1].imshow(anti_wordcloud, interpolation="bilinear")
    axarr[1,0].imshow(neutral_wordcloud, interpolation="bilinear")
    axarr[1,1].imshow(news_wordcloud, interpolation="bilinear")

    # Remove the ticks on the x and y axes
    for ax in f.axes:
        plt.sca(ax)
        plt.axis('off')

    axarr[0,0].set_title('Pro climate change\n', fontsize=35)
    axarr[0,1].set_title('Anti climate change\n', fontsize=35)
    axarr[1,0].set_title('Neutral\n', fontsize=35)
    axarr[1,1].set_title('News\n', fontsize=35)
    #plt.tight_layout()
    st.pyplot()

    st.write(pro_frequency.tail())







if option == 'Raw Tweets':
    st.subheader('Twitter Dashboard')
    #print all postive tweets
    j=1
    sortedDF = df.sort_values(by=['polarity'])
    for i in range(0, sortedDF.shape[0]):
        if(sortedDF['analysis'][i]=='Pro'):
            print(str(j)+') ' +sortedDF['message'][i])
            print()
            j = j+1


if option == 'Contact Us':
    st.subheader('Reach the App Developers here:')
    st.info('Contact details in case you any query or would like to know more of our designs:')
    st.write('Nomvuselelo Simelane: one@gmail.com')
    st.write('Thobekani Masondo: two@gmail.com')
    st.write('Ndamulelelo: three@gmail.com')
    st.write('Sandra Malope: fourl@gmail.com')
    st.write('Namhla:: five2@gmail.com')
    st.write('John: six@gmail.com')

    # Footer 
    image = Image.open('resources/imgs/EDSA_logo.png')

    st.image(image, caption='Team ZM3', use_column_width=True)

