import numpy as np
import datetime as dt
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, mean_absolute_percentage_error
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import calendar
import yfinance as yf
from multiprocessing import Pool, cpu_count
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import openai
from openai import OpenAI
from sklearn.metrics import accuracy_score, classification_report

def analyze_sentiment_gpt(text):
    if isinstance(text, float):
        text = str(text)
    score = get_gpt35_sentiment_score(text)
    score = float(score)
    return 2 if score > 0.05 else 1 if score < -0.05 else 0

# set the api key, please go to OpenAI's website and generate one, in order to run this method
openai.api_key = 'ThisIsASampleAPIKEY09431274-03428237'

def get_gpt35_sentiment_score(text):
    # command for GPT
    prompt = f"Please provide a sentiment score only with no other text\
    between -1 (very negative) and 1 (very positive) for the following text:\n\n{text}"
    chat_completion = openai.chat.completions.create(
    messages=[
        #set GPT's role
            {"role": "system", "content": "You are a sentiment analysis assistant."},
            {"role":"user",
            "content":prompt}
        ],
        # set model, e.g gpt-4, gpt-4o, gpt-3.5-turbo, etc
        model = "gpt-3.5-turbo"
    )
    sentiment_score = chat_completion.choices[0].message.content
    try:
        # return text message as float number
        return float(sentiment_score)
    except ValueError:
        # error handling if GPT is not providing a float number
        sentiment_score = re.findall("\d+\.\d+", sentiment_score)
        sentiment_score = [float(num) for num in sentiment_score]
        if len(sentiment_score) != 0:
            sentiment_score = sum(sentiment_score)/len(sentiment_score)
            return float(sentiment_score)
        else:
            return 0.0

def LSTM_model(x_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, activation="tanh", input_shape=(x_train.shape[1], len(features))))
    # set dropout rate
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, activation="tanh", return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    # set epochs and batch size
    model.fit(x_train, y_train, epochs=25, batch_size=64)
    return model



