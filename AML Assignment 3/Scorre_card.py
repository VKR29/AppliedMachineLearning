
import joblib 
import numpy as np 
from sklearn.base import BaseEstimator 

def score(text, model, threshold = 0.5):
    pred_proba = model.predict_proba([text])[0, 1]

    assert type(text) == str
    assert ((type(threshold) == float) or type(threshold) == int) and (0 <= threshold <= 1)

    if pred_proba >= threshold:
        prediction = 1
    else:
        prediction = 0
    
    return prediction, pred_proba


if __name__ == "__main__":
    model = joblib.load("AML Assignment 3/model.pkl")  # Load trained model
    text = "You have won a free lottery" 
    text = "You have to attend the meeting now"
    threshold = 0.5
    print(score(text, model, threshold))


