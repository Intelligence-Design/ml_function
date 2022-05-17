FROM python:3.8.13-bullseye as builder
WORKDIR /opt/build
COPY ./ /opt/build/
RUN --mount=type=secret,id=ssh,dst=/root/.ssh/id_rsa \
    ssh-keyscan -H github.com >> /root/.ssh/known_hosts \
    && python setup.py egg_info \
    && pip install -r *.egg-info/requires.txt
RUN --mount=type=secret,id=ssh,dst=/root/.ssh/id_rsa \
    ssh-keyscan -H github.com >> /root/.ssh/known_hosts \
    && pip install git+ssh://git@github.com/Intelligence-Design/ml_function.git@feature/first#subdirectory=ml_function_utils