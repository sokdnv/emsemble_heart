import streamlit as st
import joblib
import pandas as pd
import sklearn
from category_encoders import CatBoostEncoder

sklearn.set_config(transform_output="pandas")

ml_pipeline = joblib.load('data/ml_pipeline.pkl')

st.title('Heart disease estimator')
st.caption('Серёжи')
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('Pipeline с использованием модели **voting classifier**')
with col3:
    st.metric(label='Cross-val accuracy', value=0.8747)

st.image('data/Screenshot 2024-07-09 at 13.45.07.png')

df = pd.read_csv('data/heart.csv')

with st.sidebar:
    st.write('Вбей свои данные')
    with st.form(key='Настройки'):
        Age = st.number_input('Возраст', min_value=round(df['Age'].min()), max_value=round(df['Age'].max()), step=1)
        Sex = st.radio(label='Пол', options=['М', 'Ж'], horizontal=True)
        ChestPainType = st.radio(label='Боль в груди',
                                 options=[elem for elem in df['ChestPainType'].unique().tolist()],
                                 horizontal=True)
        RestingBP = st.number_input('Давление', min_value=80, max_value=round(df['RestingBP'].max()), step=1)
        Cholesterol = st.number_input('Холестирин', min_value=round(df['Cholesterol'].min()),
                                      max_value=round(df['Cholesterol'].max()), step=1)
        FastingBS = st.checkbox('Сахар более 120 мг/дл')
        RestingECG = st.radio(label='Электрокардиограмма',
                              options=[elem for elem in df['RestingECG'].unique().tolist()],
                              horizontal=True)
        MaxHR = st.number_input('Максимальный пульс', min_value=round(df['MaxHR'].min()),
                                max_value=round(df['MaxHR'].max()), step=1)
        ExerciseAngina = st.radio(label='Стенокардия',
                                  options=[elem for elem in df['ExerciseAngina'].unique().tolist()],
                                  horizontal=True)
        Oldpeak = st.number_input('Oldpeak', min_value=df['Oldpeak'].min(), max_value=df['Oldpeak'].max(), step=0.1)
        ST_Slope = st.radio(label='Наклон ST',
                            options=[elem for elem in df['ST_Slope'].unique().tolist()],
                            horizontal=True)

        submit = st.form_submit_button('Узнать правду!')

if submit:
    result = {
        'Age': [Age],
        'Sex': ['M' if Sex == 'М' else 'F'],
        'ChestPainType': [ChestPainType],
        'RestingBP': [RestingBP],
        'Cholesterol': [Cholesterol],
        'FastingBS': [1 if FastingBS else 0],
        'RestingECG': [RestingECG],
        'MaxHR': [MaxHR],
        'ExerciseAngina': [ExerciseAngina],
        'Oldpeak': [Oldpeak],  # Ensure non-negative value
        'ST_Slope': [ST_Slope]
    }
    df_result = pd.DataFrame(result)

    answer = ml_pipeline.predict(df_result)


    center_css = """
    <style>
        .center {
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
    </style>
    """
    cols = st.columns(3)

    st.markdown(center_css, unsafe_allow_html=True)
    if answer[0] == 1:
        with cols[1]:
            st.subheader('Могут быть проблемы с сердцем!')
        gif_path = 'https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExOWJ5YmRjYjJ5NHlxdHZ4dXFhb2pzaHJkNDAydXQwbHJuaXFjMjdrZiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/v3GUsyBpia9Zou2Kbv/giphy.webp'
    else:
        with cols[1]:
            st.subheader('Будешь жить!')
        gif_path ='https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExMWg3M2pndGZmbjFyc3V5dzN2amFoMmczcDBhazdjMDYxZjJtMHJzZSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7abKhOpu0NwenH3O/giphy.webp'

    st.markdown(f'<img src="{gif_path}" class="center" alt="GIF">', unsafe_allow_html=True)
