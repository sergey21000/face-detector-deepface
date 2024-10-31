import cv2
import numpy as np
from PIL import Image
import streamlit as st

from detector import detector_model


# ===================== Настройки страницы ======================

st.set_page_config(
    layout='wide',
    initial_sidebar_state='auto',
    page_title='Face Emotion Recognition',
    page_icon='👻',
    )


st.write("#### Детекция лиц, эмоций, пола и возраста с веб-камеры")

# ===================== Боковое меню настроек ======================

st.sidebar.header('Настройки')
st.sidebar.write('---')

face_conf_threshold = st.sidebar.slider(
    label='Порог уверенности для детекции лиц',
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.01,
    )
st.sidebar.write('---')

# какие действия нужно детектить (пока не реализовано переключение)
actions = ['age', 'gender', 'race', 'emotion']
# применять ли дополнительное выравнивание
st.sidebar.write('Применять ли дополнительное выравнивание')
align = st.sidebar.checkbox(label='Align', value=False)

# ================= Чтение видеопотока с камеры ===============

st_image = st.camera_input("Сделать фото")
st.session_state['detect_image_ready'] = False

# ====================== Детекция ===========================

if st_image:
    pil_image = Image.open(st_image)
    np_image_rgb = np.array(pil_image)
    
    np_image_bgr = cv2.cvtColor(np_image_rgb, cv2.COLOR_RGB2BGR)
    with st.spinner('Распознавание фото'):
        detections = detector_model.detect_image(
            np_image_bgr, 
            actions=actions,
            align=align,
            )

    with st.spinner('Отрисовка результата'):
        result_np_image = detector_model.draw_detections(
            np_image_rgb=np_image_rgb,
            detections=detections,
            face_conf_threshold=face_conf_threshold,
            )
        st.session_state['detect_image_ready'] = True

# ============= Отрисовка результата ======================

if st_image and st.session_state['detect_image_ready']:
    col1, col2 = st.columns(2)
    with col1:
        st.image(np_image_rgb, caption='Исходное фото', use_column_width=True)
    with col2:
        st.image(result_np_image, caption='Результат детекции', use_column_width=True)
