import numpy as np
import pickle
import pandas as pd
import streamlit as st 
from os import listdir
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from PIL import Image


pickle_in = open("classifier.pkl","rb")
classifierr=pickle.load(pickle_in)



def predict_sepsis(hr,o2,temp, SBP, MAP,DBP,Resp,BaseExcess,FiO2,pH,PaCO2, Glucose,Potassium,Hct,Age,Gender):
       
    lst=[[hr,o2,temp,SBP,MAP,DBP,Resp,BaseExcess,FiO2,pH,PaCO2,Glucose,Potassium,Hct,Age,Gender]]
    arr=np.array(lst)
    prediction=classifierr.predict(arr)
    st.subheader("Model Performance")
    plt.bar(['Accuracy','Precision','Recall','F1'],[FN,TN,TP,FP])

    return prediction



def main():
    st.title("Sepsis Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Sepsis Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    hr=int(st.slider("Heart Rate BPM",min_value=0,max_value=200,value=50))
    o2 = int(st.number_input("Oxygen Saturation(Pulse oximetry)",step=1.0))
    temp = int(st.number_input("Temperature in Celsius",step=1.0))
    SBP = int(st.number_input("Systolic BP (mm Hg)",step=1.0))
    DBP = int(st.number_input("Diastolic BP (mm Hg)",step=1.0))
    MAP = int(st.number_input("Mean arterial pressure (mm Hg)",step=1.0))
    Resp = int(st.number_input("Respiration rate (breaths per minute)",step=1.0))
    BaseExcess = int(st.number_input('BaseExcess(Measure of excess bicarbonate (mmol/L))',step=1.0))
    FiO2 = int(st.number_input("FiO2(Fraction of inspired oxygen)",step=1.0))
    pH = int(st.number_input("pH",step=1.0))
    PaCO2 = int(st.number_input("PaCO2(Partial pressure of carbon dioxide from arterial blood (mm Hg))",step=1.0))
    Glucose = int(st.number_input("Glucose(mg/dL)",step=1.0))
    Potassium = int(st.number_input("Potassium(mmol/L)",step=1.0))
    Hct = int(st.number_input("Hct(Hematocrit)",step=1.0))
    Age = int(st.number_input("Age",step=1.0)) 
    gender = st.radio("Gender",('Male','Female'))
    if gender=='Male':
         Gender=1
    else:
         Gender=0
    result_pos="Hurray!!! You do not have sepsis"
    result_neg="Alas!! You have sepsis"
    if st.button("Predict"):
        result=predict_sepsis(hr,o2, temp, SBP, MAP,DBP,Resp,BaseExcess,FiO2,pH,PaCO2, Glucose,Potassium,Hct,Age,Gender)
        if result==1:
           st.success(result_neg)
        else:
           st.success(result_pos)

if __name__=='__main__':
    main()