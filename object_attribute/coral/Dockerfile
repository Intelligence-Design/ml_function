FROM python:3.8.13-bullseye as builder

ENV TZ Asia/Tokyo
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/build
COPY ./ /opt/build/
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && apt-get update \
    && apt-get -y install libedgetpu1-std \
    && rm -rf /var/lib/apt/lists/*
RUN --mount=type=secret,id=ssh,dst=/root/.ssh/id_rsa \
    ssh-keyscan -H github.com >> /root/.ssh/known_hosts \
    && python setup.py egg_info \
    && pip install -r *.egg-info/requires.txt
RUN --mount=type=secret,id=ssh,dst=/root/.ssh/id_rsa \
    ssh-keyscan -H github.com >> /root/.ssh/known_hosts \
    && pip install git+ssh://git@github.com/Intelligence-Design/ml_function.git@feature/first#subdirectory=ml_function_utils