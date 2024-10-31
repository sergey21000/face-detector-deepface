import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import streamlit as st


plt.style.use('dark_background')
plt.rcParams.update({'figure.figsize': (7, 1.5)})
plt.rcParams.update({'font.size': 5})


def draw_plots(df: pd.DataFrame) -> None:
    df_clean = df.dropna(subset=['face_conf'])
    
    st.write('---')
    st.write('Количество детекций по возрасту и по уверенности модели')

    fig, axes = plt.subplots(1, 2)
    sns.histplot(df_clean['age'], kde=True, ax=axes[0])
    axes[0].set_title('Распределение возраста')
    axes[0].set_xlabel('Возраст')
    axes[0].set_ylabel('Количество обнаружений')
    sns.histplot(df_clean['face_conf'], kde=True, ax=axes[1])
    axes[1].set_title('Распределение уверенности детекций')
    axes[1].set_xlabel('Уверенность')
    axes[1].set_ylabel('Количество обнаружений')


    st.pyplot(fig, use_container_width=False)
    st.write('---')
    st.write('Распределение уверенности модели по классам пола и эмоций')


    fig, axes = plt.subplots(1, 2)
    sns.boxplot(
        data=df_clean,
        x='gender',
        y='face_conf',
        hue='gender',
        palette='hls',
        ax=axes[0],
        )
    axes[0].set_title('Распределение уверенности детекций по полу')
    axes[0].set_xlabel('Пол')
    axes[0].set_ylabel('Уверенность')
    # axes[0].tick_params(axis='x', labelrotation=45)
    sns.boxplot(
        data=df_clean,
        x='emotion',
        y='face_conf',
        hue='emotion',
        palette='hls',
        ax=axes[1],
        )
    axes[1].set_title('Распределение уверенности детекций по эмоциям')
    axes[1].set_xlabel('Эмоция')
    axes[1].set_ylabel('Уверенность')


    st.pyplot(fig, use_container_width=False)
    st.write('---')
    st.write('Количество детекций по классам эмоций и пола')


    fig, axes = plt.subplots(1, 2)
    sns.countplot(
        data=df_clean,
        x='emotion',
        hue='emotion',
        order=df_clean['emotion'].value_counts().index, 
        palette='viridis',
        legend=False,
        ax=axes[0],
        )
    axes[0].set_title('Количество обнаружений эмоций')
    axes[0].set_xlabel('Эмоция')
    axes[0].set_ylabel('Количество')
    sns.countplot(
        data=df_clean,
        x='gender',
        hue='gender',
        order=df_clean['gender'].value_counts().index, 
        palette='Set2',
        legend=False,
        ax=axes[1],
        )
    axes[1].set_title('Количество обнаружений пола')
    axes[1].set_xlabel('Пол')
    axes[1].set_ylabel('Количество')


    st.pyplot(fig, use_container_width=False)
    st.write('---')
    st.write('Количество детекций по классу эмоций в зависимости от пола и расы')


    fig, axes = plt.subplots(1, 2)
    sns.countplot(
        data=df_clean,
        x='emotion',
        hue='gender',
        palette='viridis',
        order=df_clean['emotion'].value_counts().index,
        ax=axes[0],
        )
    axes[0].set_title('Распределение пола по эмоциям')
    axes[0].set_xlabel('Эмоция')
    axes[0].set_ylabel('Количество')
    axes[0].legend(title='Пол')
    sns.countplot(
        data=df_clean,
        x='emotion',
        hue='race',
        palette='viridis',
        order=df_clean['emotion'].value_counts().index,
        ax=axes[1],
        )
    axes[1].set_title('Распределение рас по эмоциям')
    axes[1].set_xlabel('Эмоция')
    axes[1].set_ylabel('Количество')
    axes[1].legend(title='Раса')


    st.pyplot(fig, use_container_width=False)
    st.write('---')
    st.write('Кадры и секунды с наибольшим количеством детекций')


    fig, axes = plt.subplots(1, 2)
    face_count_per_frame = df.groupby('frame_num')['face_detected'].sum()
    axes[0].plot(face_count_per_frame.index, face_count_per_frame.values, marker='o', linestyle='-')
    axes[0].set_title('Частота обнаружения лиц по кадрам')
    axes[0].set_xlabel('Номер кадра')
    axes[0].set_ylabel('Количество обнаруженных лиц')
    face_count_per_frame = df.groupby('frame_sec')['face_detected'].sum()
    axes[1].plot(face_count_per_frame.index, face_count_per_frame.values, marker='o', linestyle='-')
    axes[1].set_title('Частота обнаружения лиц по секундам')
    axes[1].set_xlabel('Время (сек)')
    axes[1].set_ylabel('Количество обнаруженных лиц')


    st.pyplot(fig, use_container_width=False)
    st.write('---')
    st.write('Какие классы в какое время были обнаружены (по эмоциям)')


    fig, ax = plt.subplots(1, 1)
    sns.scatterplot(
        data=df_clean, 
        x='frame_sec', 
        y='emotion', 
        hue='emotion', 
        palette='deep', 
        s=50, 
        alpha=0.6, 
        legend=True,
        ax=ax,
        )
    ax.set_title('Временная шкала обнаружения лиц по эмоциям')
    ax.set_xlabel('Время видео (секунды)')
    ax.set_ylabel('Эмоция')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(title='Классы эмоций', bbox_to_anchor=(1.05, 1), loc='upper left')


    st.pyplot(fig, use_container_width=False)
    st.write('---')
    st.write('Какие классы в какое время были обнаружены (по полу)')


    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df_clean, 
        x='frame_sec', 
        y='gender', 
        hue='gender', 
        palette='deep', 
        s=50, 
        alpha=0.6, 
        legend=True,
        ax=ax,
        )
    ax.set_title('Временная шкала обнаружения лиц по полу')
    ax.set_xlabel('Время видео (секунды)')
    ax.set_ylabel('Пол')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(title='Классы пола', bbox_to_anchor=(1.05, 1), loc='upper left')


    st.pyplot(fig, use_container_width=False)
    st.write('---')
    st.write('Какие классы в какое время были обнаружены (по расе)')


    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df_clean, 
        x='frame_sec', 
        y='race', 
        hue='race', 
        palette='deep', 
        s=50, 
        alpha=0.6, 
        legend=True,
        ax=ax,
        )
    ax.set_title('Временная шкала обнаружения лиц по расе')
    ax.set_xlabel('Время видео (секунды)')
    ax.set_ylabel('Раса')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(title='Классы расы', bbox_to_anchor=(1.05, 1), loc='upper left')


    st.pyplot(fig, use_container_width=False)
    st.write('---')
    st.write('Распределение смены эмоций по времени и кол-ву детекций')


    fig, ax = plt.subplots()
    emotion_timeline = df.pivot_table(
        index='frame_sec', 
        columns='emotion', 
        aggfunc='size', 
        fill_value=0,
        )
    emotion_timeline.plot(kind='area', stacked=True, ax=ax)
    ax.set_title('Изменение эмоций во времени')
    ax.set_xlabel('Время видео (секунды)')
    ax.set_ylabel('Количество детекций')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(title='Эмоции', bbox_to_anchor=(1.05, 1), loc='upper left')


    st.pyplot(fig, use_container_width=False)
    st.write('---')


