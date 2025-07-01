import streamlit as st
import joblib
import pandas as pd

spam_model=joblib.load('spam_classifier.pkl')
language_model=joblib.load('lang_det.pkl')
news_model=joblib.load('news_cat.pkl')
review_model=joblib.load('review.pkl')


st.set_page_config(layout='wide')

st.markdown("""
    <h1 style='text-align: center; color: #9B59B6;'>
        ✨💡 LENS eXpert 💡✨ <br>
        <span style='font-size: 24px; color: #566573;'>(NLP Suits)</span>
    </h1>
    """, unsafe_allow_html=True)

tab1,tab2,tab3,tab4=st.tabs(['✉️ Spam Classifier','🗣️ Language Dectection','🍔 Food Review Sentiment','🗞️ News Classification'])

with tab1:
    st.header('✉️ Spam Classifier')
    st.write('A Spam Classifier is a system that automatically identifies whether a message is spam (unwanted or harmful content) or ham (legitimate content). It helps in filtering out irrelevant or potentially dangerous messages like ads, phishing links, or fraud attempts.')
    msg1=st.text_input('Enter Msg',key='msg1')
    if st.button('Prediction',key='b1'):
        pre=spam_model.predict([msg1])
        if pre[0]==0:
            st.image('D:/DataScience/ML viideo/New folder/spam.jpg')
        else:
            st.image('D:/DataScience/ML viideo/New folder/not_spam.pn g')
        
    uploader_file1=st.file_uploader('upload file containing bulk msg',type=['csv','txt'],key='uploader_file1')
    
    if uploader_file1:
        df_spam=pd.read_csv(uploader_file1,header=None,names=['Msg'])  

        pred=spam_model.predict(df_spam.Msg)
        df_spam.index=range(1,df_spam.shape[0]+1)
        df_spam['Prediction']=pred
        df_spam['Prediction']=df_spam['Prediction'].map({0:'Spam',1:'Not Spam'})
        st.dataframe(df_spam)

with tab2:
    st.header('🗣️ Language Detection ')
    st.write('Language Detection is the process of automatically identifying the language in which a given piece of text is written. This is a crucial first step in many Natural Language Processing (NLP) pipelines, especially when working with multilingual data.')
    msg2=st.text_input('Enter Msg',key='msg2')
    if st.button('Prediction',key='b2'):
        pre=language_model.predict([msg2])
        st.success(pre)
        
    uploader_file2=st.file_uploader('upload file containing bulk msg',type=['csv','txt'],key='uploader_file2')
    
    if uploader_file2:
        df_lang=pd.read_csv(uploader_file2,header=None,names=['Msg'])  

        pred=language_model.predict(df_lang.Msg)
        df_lang.index=range(1,df_lang.shape[0]+1)
        df_lang['Prediction']=pred
        df_lang['Prediction']=df_lang['Prediction']
        st.dataframe(df_lang)

with tab3:
    st.header('🍔 Food Review Sentiment')
    st.write('Food Review Sentiment Analysis automatically determines whether a food review expresses a positive, negative, or neutral opinion. It helps businesses and users quickly understand public sentiment toward food, restaurants, or services.')
    msg3=st.text_input('Enter Msg',key='msg3')
    if st.button('Prediction',key='b3'):
        pre=review_model.predict([msg3])
        if pre[0]==0:
            st.warning('😞 Not Satisfied')
        else:
            st.success('😃 Satisfied')
        
    uploader_file3=st.file_uploader('upload file containing bulk msg',type=['csv','txt'],key='uploader_file3')
    
    if uploader_file3:
        df_review=pd.read_csv(uploader_file3,header=None,names=['Msg'])  

        pred=review_model.predict(df_review.Msg)
        df_review.index=range(1,df_review.shape[0]+1)
        df_review['Prediction']=pred
        df_review['Prediction']=df_review['Prediction'].map({0:'😞 Not Satisfied',1:'😃 Satisfied'})
        st.dataframe(df_review)

with tab4:
    st.image('download.jpeg')

st.sidebar.image('robot.webp')
with st.sidebar.expander('ℹ️ About us'):
    st.write('We are a group of students trying to understand the concept of NLP')
    st.write('''Welcome to 🔍 LENS eXpert (NLP Suits) — your all-in-one Natural Language Processing toolkit designed to make text analysis easy, fast, and smart!

Our platform brings together four powerful NLP solutions under one roof to help you efficiently process and understand language data in the real world.

✨ Our Modules:
✉️ Spam Classifier
We protect your inbox and communication channels by accurately detecting and filtering out unwanted spam messages. Whether it's emails, SMS, or social media comments, our spam classifier keeps your space clean and safe.

🗣️ Language Detection
With just a few words, our system quickly detects the language of your text, enabling seamless processing of multilingual content. Perfect for global businesses, chatbots, and multilingual platforms.

🍔 Food Review Sentiment
Discover what people really think about your food! Our sentiment analysis tool scans customer reviews and determines whether they are positive, negative, or neutral, helping restaurants and businesses improve their services and offerings.

🗞️ News Classification
Stay organized and informed with our news classification module. It automatically categorizes news articles by topics, helping you sort and process vast amounts of information quickly and accurately.''')

with st.sidebar.expander('🌐 Contact'):
    st.write('📞 9999999999')
    st.write('📧 guptasubham@797gmail.com')

with st.sidebar.expander('Help'):
    st.write('subham')
