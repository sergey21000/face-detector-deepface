import io
import cv2
import numpy as np
from PIL import Image
import streamlit as st


# ===================== Настройки страницы ======================

st.set_page_config(
    layout='wide',
    initial_sidebar_state='auto',
    page_title='Face Emotion Recognition',
    page_icon='👻',
    )

st.write("#### Детекция лиц, эмоций, пола, расы и возраста на изображении")

# отображение картинки с примерами детекции
@st.cache_data
def load_main_image(image_path: str) -> Image.Image:
    main_pil_image = Image.open(image_path)
    return main_pil_image

MAIN_IMAGE_PATH = './media/emotions_detect.png'
main_pil_image = load_main_image(MAIN_IMAGE_PATH)
st.image(main_pil_image, width=800)

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


# загрузка и инициализация моделей
with st.spinner('Инициализация/загрузка моделей...'):
    from detector import detector_model

# ==================== Загрузка изображения и распознавание ============

st_image = st.file_uploader(label='Выберите изображение')
st.session_state['detect_image_ready'] = False

if st_image:
    pil_image = Image.open(st_image)
    np_image_rgb = np.array(pil_image)
    
    np_image_bgr = cv2.cvtColor(np_image_rgb, cv2.COLOR_RGB2BGR)
    st.image(np_image_bgr, width=400, channels='BGR')
    
    # ================ Кнопка распознать ==================    

    if st.button('Распознать'):
        if detector_model.is_first_run:
            spinner_text = 'Первоначальная инициализация моделей и распознавание фото...'
        else:
            spinner_text = 'Распознавание фото ...'

        with st.spinner(spinner_text):
            detections = detector_model.detect_image(
                np_image_bgr, 
                actions=actions,
                align=align,
                )
            detector_model.is_first_run = False

        with st.spinner('Отрисовка результата'):
            result_np_image = detector_model.draw_detections(
                np_image_rgb=np_image_rgb,
                detections=detections,
                face_conf_threshold=face_conf_threshold,
                )
            st.session_state['detect_image_ready'] = True

# отображение результата детекции
if st.session_state['detect_image_ready']:
    st.image(result_np_image)

    # сохранение картинки с результатом чтобы ее скачать
    image_name = f"{st_image.name.split('.')[0]}_detect.png"
    file_buffer = io.BytesIO()
    result_pil_image = Image.fromarray(result_np_image)
    result_pil_image.save(file_buffer, format='PNG')
    image_bytes = file_buffer.getvalue()
    
    # ================ Кнопка скачать ================== 

    st.download_button(
        label='Скачать изображение', 
        data=image_bytes, 
        file_name=image_name,
        mime='image/png',
        )
    st.session_state['detect_image_ready'] = False
