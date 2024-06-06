import streamlit as st
import nltk

# Download the punkt tokenizer if not already present
nltk.download('punkt')

st.set_page_config(
    page_title="NLP WEB APP",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# NLP packages
from textblob import TextBlob
import spacy
import neattext as nt
from collections import Counter
import re
from deep_translator import GoogleTranslator
# Viz packages
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from wordcloud import WordCloud
# Extra packages
import time
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

def summarize_text(text, num_sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    summary_text = " ".join([str(sentence) for sentence in summary])
    return summary_text

@st.cache_data
# Lemma and Tokens Function
def text_analyzer(text):
    # Importing English Library
    nlp = spacy.load('en_core_web_sm')
    # Creating an nlp object
    doc = nlp(text)
    # Extracting Tokens and Lemmas
    allData = [('"Token":{},\n"Lemma":{}'.format(token.text, token.lemma_)) for token in doc]
    return allData

def main():
    """NLP web app using streamlit"""

    title_template = """
    <div style="background-color:#333333; padding:5px;">
    <h1 style="color:cyan">NLP WEB APP</h1>
    </div>
    """
    st.markdown(title_template, unsafe_allow_html=True)

    st.sidebar.image("NLP-examples-scaled.jpeg", use_column_width=True)

    activity = ["Text Analysis", "Translation", "Sentiment Analysis", "About"]
    choice = st.sidebar.selectbox("Menu", activity)
    if choice == "Text Analysis":
        st.subheader('Text Analysis')
        raw_text = st.text_area("Enter your text in English", height=250)
        if st.button("Analyze"):
            if len(raw_text) == 0:
                st.warning("Error! No Text To Analyze")
            else:
                blob = TextBlob(raw_text)
                my_bar = st.progress(0)
                message_slot = st.empty()
                message_slot.text('Analyzing...')

                for value in range(50):
                    time.sleep(0.01)
                    my_bar.progress(value + 1)

                if my_bar.progress(100):
                    message_slot.empty()  # Clear the "Generating..." message
                    st.success("Your Text Has Been Analyzed successfully!")
                st.info("Basic Function")
                col1, col2 = st.columns(2)
                with col1:
                    with st.expander("Basic info"):
                        st.write("Text Stats")
                        word_desc = nt.TextFrame(raw_text).word_stats()
                        result_desc = {"Length of Text": word_desc['Length of Text'],
                                       "Num of Vowels": word_desc['Num of Vowels'],
                                       "Num of Consonants": word_desc['Num of Consonants'],
                                       "Num of Stopwords": word_desc['Num of Stopwords']}
                        st.write(result_desc)
                    with st.expander("Stopwords"):
                        st.success("Stop Words list")
                        stop_w = nt.TextExtractor(raw_text).extract_stopwords()
                        st.error(stop_w)

                with col2:
                    with st.expander("Processed Text"):
                        st.success("Stopwords Excluded Text")
                        processed_text = str(nt.TextFrame(raw_text).remove_stopwords())
                        st.write(processed_text)
                    with st.expander("Plot WordCloud"):
                        st.success("WordCloud")
                        Wordcloud = WordCloud().generate(processed_text)
                        fig = plt.figure(1, figsize=(20, 10))
                        plt.imshow(Wordcloud, interpolation='bilinear')
                        plt.axis('off')
                        st.pyplot(fig)
                st.info("Advanced Features")
                col3, col4 = st.columns(2)
                with col3:
                    with st.expander("Tokens & Lemmas"):
                        st.write("T&K")
                        processed_text_mid = str(nt.TextFrame(raw_text).remove_stopwords())
                        processed_text_mid = str(nt.TextFrame(processed_text_mid).remove_puncts())
                        processed_text_fin = str(nt.TextFrame(processed_text_mid).remove_special_characters())
                        tandl = text_analyzer(processed_text_fin)
                        st.json(tandl)
                with col4:
                    with st.expander("Summarize"):
                        st.success("Summarized Text is as Follows")
                        summary = summarize_text(raw_text)
                        st.info(summary)

    if choice == "Translation":
      st.subheader('Text Translation')
      st.write("")
      raw_text = st.text_area("Write Something to be Translated", height=250)
      
      target_lang = st.selectbox("Choose Language", ["German", "Spanish", "French", "Hindi"])
      display_target_lang=target_lang
      if target_lang == "German":
          target_lang = 'de'
      elif target_lang == "Spanish":
          target_lang = 'es'
      elif target_lang == "French":
          target_lang = 'fr'
      else:
          target_lang = 'hi'
      
      if st.button("Translate"):
          if len(raw_text) < 3:
              st.warning("Sorry! You need to provide a text with at least 3 characters")
          else:
              my_bar = st.progress(0)
              message_slot = st.empty()
              message_slot.text('Translating...')

              for value in range(50):
                  time.sleep(0.01)
                  my_bar.progress(value + 1)

              if my_bar.progress(100):
                  message_slot.empty()  # Clear the "Translating..." message
                  st.success("Your Text Has Been Translated successfully!")
                  
              translator = GoogleTranslator(source="auto", target=target_lang)
              translated_text = translator.translate(raw_text)
              st.info("Original text: "+raw_text)
              st.warning("language selected is: "+display_target_lang)
              st.info("Translated text: "+translated_text)
    if choice == "Sentiment Analysis":
      st.subheader('Sentiment Analysis')
      st.write("")
      raw_text = st.text_area("Enter text to analyze", height=200)
      if st.button("Evaluate"):
          if len(raw_text) == 0:
              st.warning("Sorry! You have not entered any Text")
          else:
              blob = TextBlob(raw_text)
              polarity = blob.sentiment.polarity
              subjectivity = blob.sentiment.subjectivity

              # Determine polarity as positive or negative
              if polarity > 0:
                  polarity_label = "Positive"
              elif polarity < 0:
                  polarity_label = "Negative"
              else:
                  polarity_label = "Neutral"

              # Determine subjectivity type
              if subjectivity > 0.5:
                  subjectivity_type = "Personal"
              else:
                  subjectivity_type = "Factual"

              my_bar = st.progress(0)
              message_slot = st.empty()
              message_slot.text('Evaluating...')

              for value in range(50):
                  time.sleep(0.01)
                  my_bar.progress(value + 1)

              if my_bar.progress(100):
                  message_slot.empty()  # Clear the "Evaluating..." message
                  st.success("Your Text Has Been Evaluated successfully!")
              
              st.info("Sentiment Analysis")
              st.write("Polarity:", polarity_label)
              st.write("Subjectivity:", subjectivity_type)
              st.write("")
    if choice == "About":
     st.subheader('About')
     st.write("Welcome to the NLP Web App!")
     introduction = """
     Welcome to the future of text analysis with our state-of-the-art NLP Web App! In the digital age where information flows ceaselessly, understanding and processing text data efficiently has become indispensable. Our app harnesses the power of advanced Natural Language Processing (NLP) techniques to transform raw text into meaningful insights. Whether you're a researcher, a content creator, or simply a language enthusiast, our app provides a suite of tools designed to enhance your text processing capabilities with ease and precision.
     """
     feature_description = """
     Our NLP Web App offers a rich array of features tailored to meet your diverse needs. Dive into the "Text Analysis" section, where you can input any English text and receive a detailed breakdown of its linguistic components, including tokenization, lemmatization, and word statistics. Generate insightful word clouds to visualize the most prominent terms and filter out stopwords to refine your analysis. For multilingual users, the "Translation" feature leverages the robust Google Translator to convert your text into various languages such as German, Spanish, French, and Hindi, ensuring seamless communication across linguistic boundaries. Additionally, our "Sentiment Analysis" tool evaluates the emotional tone of your text, providing a clear categorization of its polarity and subjectivity, empowering you to gauge the sentiment behind the words.
     """
     closing_statement = """
     Step into a world where text is more than just words with our NLP Web App. Designed with both simplicity and sophistication in mind, our app opens the door to a deeper understanding of textual data. Whether you're analyzing the sentiment of customer reviews, summarizing lengthy articles, or exploring the nuances of language, our NLP Web App is your trusted companion. We invite you to explore its features, discover its capabilities, and elevate your text analysis experience. Should you have any suggestions or require assistance, our team is here to support you every step of the way. Welcome to the future of text analysis, where every word matters.
     """   
     st.write(introduction)
     st.write(feature_description)
     st.write(closing_statement)
     st.write("For collaboration, issue reporting, or feature suggestions, you can contact me at: abhirupbasu30@gmail.com")
if __name__ == "__main__":
    main()