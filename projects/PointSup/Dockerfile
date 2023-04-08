# ベースイメージとして公式のPythonイメージを使用
FROM python:3.9-slim-buster

# 必要なツールとライブラリをインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# 必要なパッケージをインストール
RUN /usr/local/bin/python -m pip install --upgrade pip
# まず、torch, torchvision, streamlit, opencv-python, pyngrokをインストールする
RUN pip install --no-cache-dir streamlit opencv-python pyngrok
RUN pip install --no-cache-dir torch torchvision -f https://download.pytorch.org/whl/cu111/torch_stable.html
# 次に、detectron2をインストールする
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Streamlitのアプリファイルおよびその他の必要なファイル(モデルやフォルダ) をコンテナにコピー
COPY app.py /app/
COPY point_sup /app/point_sup
COPY model /app/model
COPY configs /app/config

# ワーキングディレクトリを設定
WORKDIR /app

# Streamlitを実行するコマンドを指定
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
