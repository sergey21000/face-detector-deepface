FROM ghcr.io/sergey21000/face-detector-deepface:main-cpu

RUN useradd -m -u 1000 user \
    && chown -R user:user /app

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

CMD ["streamlit", "run", "Image_Detection.py", \
    "--server.enableXsrfProtection=false", \
    "--server.enableCORS=false", \
    "--server.showEmailPrompt=False", \
    "--server.port=8501", \
    "--server.address=0.0.0.0"]