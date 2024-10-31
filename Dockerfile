FROM tensorflow/tensorflow:2.15.0-gpu
RUN apt-get update && apt-get install -y ffmpeg
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --ignore-installed blinker==1.8.2
RUN pip install --no-cache-dir $(grep -ivE "tensorflow|tf-keras|--extra-index-url" requirements.txt)
COPY media/ ./media/
COPY fonts/ ./fonts/
COPY pages/ ./pages/
COPY detector.py Image_Detection.py draw_utils.py .
EXPOSE 8501
CMD ["streamlit", "run", "Image_Detection.py", "--server.port=8501", "--server.address=0.0.0.0"]
