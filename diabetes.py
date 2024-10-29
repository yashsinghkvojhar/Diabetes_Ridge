import pandas as pd
import streamlit as st
import pickle
diabetes_df=pd.read_csv("./diabetes.csv")

st.write(
    '''
    
    # Diabetes Check
    '''
)

st.dataframe(diabetes_df.head())
encode_dict={
    'sex':{'Male':0.050680,'Female':-0.044642}
}

col1,col2,col3,col4,col5=st.columns(5   )

age=col1.slider("Select the age",0,100,step=1)

sex=col1.selectbox("Select the gender",['Male','Female'])

bmi=col2.slider("Select the bmi",-0.09,0.17, step=0.01)
bp=col2.slider("Select the blood pressure",-0.11,0.13, step=0.01)
s1=col3.slider("Select total serum cholestrol",-0.12,0.15, step=0.01)
s2=col3.slider("Select Low-density lipoproteins",-0.11,0.19, step=0.01)
s3=col4.slider("Select High-density lipoproteins",-0.10,0.18, step=0.01)
s4=col4.slider("Select total cholestrol",-0.07,0.18, step=0.01)
s5=col5.slider("Select possibly log of serum triglycerides level",-0.12,0.13, step=0.01)
s6=col5.slider("Select Blood sugar level",-0.13,0.13, step=0.01)

def model_pred(age,encoded_sex,bmi,bp,s1,s2,s3,s4,s5,s6):
    loaded_model=pickle.load(open('diabetes_model.pkl','rb'))
    return loaded_model.predict([[age,encoded_sex,bmi,bp,s1,s2,s3,s4,s5,s6]])

if st.button("Predict Diabetes"):
    encoded_sex=encode_dict['sex'][sex]

    sugar=model_pred(age,encoded_sex,bmi,bp,s1,s2,s3,s4,s5,s6)

    st.header("Diabetes Prediction"+str(sugar))