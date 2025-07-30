import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib



try:
    df = pd.read_csv("./datasets/ClaimFakeCOVID-19_5.csv")
    print("Dataset loaded successfully")

    df = df[['fact_check_url','news_url','title']]
    # it removes all the rows which have missing values to remove consistency and ensure data quality
    df.dropna(inplace=True)
    # for printing the first 5 rows of the dataset
    print(df.head())




except FileNotFoundError:
    print("file not found")