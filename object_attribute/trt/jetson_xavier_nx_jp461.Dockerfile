FROM nvcr.io/nvidia/l4t-tensorrt:r8.2.1-runtime as builder

ENV TZ Asia/Tokyo
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/build
COPY ./ /opt/build/
RUN apt-get update && \
    apt-get install -y openssh-server \
    build-essential \
    cmake \
    python3.8-dev
RUN --mount=type=secret,id=ssh,dst=/root/.ssh/id_rsa \
    ssh-keyscan -H github.com >> /root/.ssh/known_hosts \
    && python setup.py egg_info \
    && pip install -r *.egg-info/requires.txt
RUN --mount=type=secret,id=ssh,dst=/root/.ssh/id_rsa \
    ssh-keyscan -H github.com >> /root/.ssh/known_hosts \
    && pip install git+ssh://git@github.com/Intelligence-Design/ml_function.git@feature/first#subdirectory=ml_function_utils