import tempfile

import streamlit as st


# ===================== Настройки страницы ======================

st.set_page_config(
    layout='wide',
    initial_sidebar_state='auto',
    page_title='Face Emotion Recognition',
    page_icon='👻',
    )

st.write("#### Детекция лиц, эмоций, пола, расы и возраста на видео")

# загрузка и отображение видео с примером детекции
@st.cache_data 
def load_main_video(video_path: str) -> bytes:
    with open(video_path, 'rb') as file:
        video_bytes = file.read()
    return video_bytes

MAIN_VIDEO_PATH = 'media/result_video.mp4'
example_video_bytes = load_main_video(MAIN_VIDEO_PATH)

video_width = 60
video_side = 100

_, container, _ = st.columns([video_side, video_width, video_side])
container.video(data=example_video_bytes)

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

# ==================== Загрузка видео и детекция =================

st_video = st.file_uploader(label='Выберите видео')
st.session_state['video_ready_to_convert'] = False
st.session_state['annotations_ready'] = False

if st_video:
    if st.button('Детекция видео'):
        progress_text = 'Детекция видео...'
        progress_bar = st.progress(0, text=progress_text)

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(st_video.read())
        video_file = temp_file.name

        generator = detector_model.detect_video(
            video_file=video_file,
            actions=actions,
            align=align,
            face_conf_threshold=face_conf_threshold,
            )
        frame_count, total_frames = next(generator)
        for (frame_count, _) in generator:
            progress_text = f'Детекция видео, кадр {frame_count}/{total_frames}'
            progress_bar.progress(frame_count / total_frames, text=progress_text)
        
        progress_bar.empty()
        st.session_state['video_ready_to_convert'] = True
        detector_model.detections_to_df()

# ======= Конвертация видео для отображения в браузере =================

if st.session_state['video_ready_to_convert']:
    convert_video_path = 'result_video_convert.mp4'
    with st.spinner('Идет конвертация видео ...'):
        detector_model.convert_mp4(detector_model.save_video_path, convert_video_path)
    # st.video(convert_video_path, format='video/mp4')

    with open(str(convert_video_path), 'rb') as file:
        video_bytes = file.read()

    _, container, _ = st.columns([video_side, video_width, video_side])
    container.video(data=video_bytes)
    
    # ======================= Кнопка скачать видео ===================

    st.download_button(
        label='Скачать видео', 
        data=video_bytes, 
        file_name=detector_model.save_video_path,
        )
    st.session_state['video_ready_to_convert'] = False

