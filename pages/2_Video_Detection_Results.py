from pathlib import Path

import pandas as pd
import streamlit as st

from draw_utils import draw_plots


st.set_page_config(
    layout='wide',
    initial_sidebar_state='auto',
    page_title='Face Emotion Recognition',
    page_icon='👻',
    )


st.write("#### Отображение результатов детекции видео")

csv_path = 'video_annotations.csv'

@st.cache_data
def load_df():
    df = pd.read_csv(csv_path)
    return df

if Path('video_annotations.csv').exists():
    df = load_df()
    st.download_button(
        label='Скачать csv аннотации', 
        data=df.to_csv().encode('utf-8'), 
        file_name=csv_path,
        mime='text/csv',
        )
    draw_plots(df)
else:
    st.write(
        '#### Для отображения результатов детекции выберите видео ' \
        'на вкладке Video Detection и нажмите "Детекция видео"'
    )