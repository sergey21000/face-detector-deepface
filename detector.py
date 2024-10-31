import os
from pathlib import Path
from dataclasses import dataclass
from typing import Generator

import cv2
import numpy as np
import pandas as pd

from PIL import Image, ImageDraw, ImageFont
from ffmpeg import FFmpeg

# установка пути куда будут скачиваться модели DeepFace
MODELS_DIR = Path('models')
os.environ['DEEPFACE_HOME'] = str(MODELS_DIR)


# настройка Tensorflow чтобы не вылетало если мало памяти
import tensorflow as tf

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as ex:
        print(ex)


from deepface import DeepFace


# словарь для детекции одного лица который возвращает библиотека DeepFace
DEEPFACE_DICT = dict[str, str | int | float | dict[str, str | int | float | None]]

# датакласс для хранения результатов детекции
# если лиц не обнаружено (face_conf = 0) то устанавливает все значения на None
@dataclass
class DetectResult:
    face_conf: float | None = None
    gender: str | None = None
    gender_conf: float | None = None
    age: int | None = None
    race: str | None = None
    race_conf: float | None = None
    emotion: str | None = None
    emotion_conf: float | None = None
    x: int | None = None
    y: int | None = None
    w: int | None = None
    h: int | None = None


    # если лиц не обнаружено (face_conf = 0) то устанавливает все значения на None
    def __post_init__(self):
        if self.face_conf == 0:
            self.__dict__.update(DetectResult().__dict__)


    # формирование экземпляра из словаря, который возвращает библиотека DeepFace
    @classmethod
    def from_dict(cls, detect_dict: DEEPFACE_DICT):
        return cls(
            face_conf=detect_dict['face_confidence'],
            gender=detect_dict['dominant_gender'],
            gender_conf=detect_dict['gender'][detect_dict['dominant_gender']],
            age=detect_dict['age'],
            race=detect_dict['dominant_race'],
            race_conf=detect_dict['race'][detect_dict['dominant_race']],
            emotion=detect_dict['dominant_emotion'],
            emotion_conf=detect_dict['emotion'][detect_dict['dominant_emotion']],
            x=detect_dict['region']['x'],
            y=detect_dict['region']['y'],
            w=detect_dict['region']['w'],
            h=detect_dict['region']['h'],
        )


    # вернуть текущий экземпляр в качестве словаря
    def as_dict(self):
        return self.__dict__


    # получить названия атрибутов экземпляра
    def keys(self):
        return self.__dict__.keys()


    # видоизменить текстовое представление экземпляра
    def __repr__(self):
        if self.face_conf is None:
            repr_text = f'DetectResult(face_conf=None'
        else:
            repr_text = 'DetectResult(\n  ' + \
                '\n  '.join([f'{k}={v}' for k, v in self.as_dict().items()])
        return repr_text + '\n)\n'


class Detector:
    def __init__(self):
        self.box_color = 'red'
        self.text_color = 'yellow'
        self.fill_color = 'black'
        self.font_path = 'fonts/LiberationMono-Regular.ttf'
        self.font_scale = 40  # чем меньше тем больше шрифт
        self.detections_all_frames = []
        self.save_video_path = 'result_video.mp4'
        self.result_csv_path = 'video_annotations.csv'
        # opencv, yunet, centerface, dlib, ssd, fastmtcnn
        self.detector_backend = 'ssd'
        weights_dir = MODELS_DIR / '.deepface' / 'weights'
        self.is_first_run = True

        if Path(self.result_csv_path).is_file():
            Path(self.result_csv_path).unlink(missing_ok=True)

        if len(list(weights_dir.iterdir())) == 0:
            self.first_detect_and_load_weights()


    # детекция случайной картинки из шума чтобы при первом запуске загрузились веса
    def first_detect_and_load_weights(self):
        DeepFace.analyze(
            img_path=np.random.randint(0, 256, size=(28, 28, 3), dtype=np.uint8),
            actions=['age', 'gender', 'race', 'emotion'],
            detector_backend=self.detector_backend,
            align=False,
            enforce_detection=False,
            silent=True,
            )


    # детекция одного изображения, возаращает список со словарями результатов детекций
    def detect_image(self, image: str | np.ndarray, actions: list[str], align: bool) -> list[DetectResult]:
        detection_dicts = DeepFace.analyze(
            img_path=image,
            actions=actions,
            detector_backend=self.detector_backend,
            align=align,
            enforce_detection=False,
            silent=True,
            )
        detections = [DetectResult.from_dict(detection) for detection in detection_dicts]
        return detections


    # отрисовка результатов для детекций
    def draw_detections(
            self,
            np_image_rgb: np.ndarray,
            detections: list[DetectResult],
            face_conf_threshold: float,
            ) -> np.ndarray:
        pil_image = Image.fromarray(np_image_rgb)
        draw = ImageDraw.Draw(pil_image)
        font_size = pil_image.size[1] // self.font_scale
        # font = ImageFont.load_default(size=font_size)
        font = ImageFont.truetype(self.font_path, size=font_size)
        for detection in detections:
            if detection.face_conf is not None:
                if detection.face_conf > face_conf_threshold:
                    x, y, w, h = detection.x, detection.y, detection.w, detection.h
                    text = f'{detection.gender},{detection.age},{detection.race},{detection.emotion}'
                    draw.rectangle((x, y, x + w, y + h), outline=self.box_color, width=6)
                    _, _, text_width, text_height = font.getbbox(text)
                    if (x + text_width) > pil_image.size[0]:
                        x -= text_width / 2
                    draw.rectangle(xy=(x, y - text_height, x + text_width, y), fill=self.fill_color)
                    draw.text(xy=(x, y), text=text, font=font, fill=self.text_color, anchor='lb')
        return np.array(pil_image)


    # функция - генератор для детекция видео, который детектит кадры видео в цикле
    # возвращает номер текущего кадра и общее кол-во кадров, чтобы сделать внешний прогресс бар
    def detect_video(
            self,
            video_file: str | Path,
            actions: list[str],
            align: bool,
            face_conf_threshold: float,
            ) -> Generator[tuple[int, int], None, None]:

        cap_read = cv2.VideoCapture(str(video_file))
        frames_width = int(cap_read.get(cv2.CAP_PROP_FRAME_WIDTH))
        frames_height = int(cap_read.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_fps = int(cap_read.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap_read.get(cv2.CAP_PROP_FRAME_COUNT))

        cap_write = cv2.VideoWriter(
            filename=self.save_video_path,
            fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
            fps=frames_fps,
            frameSize=(frames_width, frames_height),
            )

        self.detections_all_frames = []
        frames_count = 0

        while cap_read.isOpened():
            ret, np_image_bgr = cap_read.read()
            if not ret:
                break
            detections = self.detect_image(
                image=np_image_bgr,
                actions=actions,
                align=align,
                )
            self.detections_all_frames.append(detections)

            if detections[0].face_conf is not None:
                np_image_rgb = cv2.cvtColor(np_image_bgr, cv2.COLOR_BGR2RGB)
                result_np_image = self.draw_detections(
                    np_image_rgb=np_image_rgb,
                    detections=detections,
                    face_conf_threshold=face_conf_threshold,
                )
                np_image_bgr = cv2.cvtColor(result_np_image, cv2.COLOR_RGB2BGR)

            cap_write.write(np_image_bgr)
            frames_count += 1
            yield (frames_count, total_frames)

        cap_write.release()
        cap_read.release()


    # создание датафрейма из словарей с результатами детекции
    # каждая строка датафрейма - порядковый номер кадра из видео
    # если на кадре ничего нет то все значения кроме номера кадра и времени видео будут None
    # если на кадре несколько лиц - номера кадров и время видео будут дублироваться
    def detections_to_df(self) -> pd.DataFrame:
        if not Path(self.save_video_path).exists():
            print('Нет видео с результатами детекции')
            return None

        if len(self.detections_all_frames) == 0:
            print('Не найдено ни одного объекта')
            return None

        cap_read = cv2.VideoCapture(str(self.save_video_path))
        frames_fps = int(cap_read.get(cv2.CAP_PROP_FPS))
        cap_read.release()

        df_list = []
        for frame_num, detections in enumerate(self.detections_all_frames, start=1):
            for detection in detections:
                df_list.append({'frame_num': frame_num, **detection.as_dict()})

        df = pd.DataFrame(df_list)
        df.insert(loc=1, column='frame_sec', value=df.frame_num / frames_fps)
        df['face_detected'] = df['gender'].notna().astype(int)
        df.to_csv(self.result_csv_path, index=False)
        return self.result_csv_path


    # конвертация в mp4
    @staticmethod
    def convert_mp4(input_video_path: str | Path, output_video_path: str | Path) -> None:
        ffmpeg = FFmpeg().option('y').input(input_video_path).output(output_video_path)
        ffmpeg.execute()

detector_model = Detector()
