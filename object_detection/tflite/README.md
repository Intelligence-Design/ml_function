# Online install

```
pip install git+ssh://git@github.com/Intelligence-Design/ml_function.git@feature/first#subdirectory=object_detection/tflite
```

# Docker image build

```
DOCKER_BUILDKIT=1 docker build --secret id=ssh,src=${HOME}/.ssh/id_rsa -t object_detection_tflite .
```

# Docker container run

```
docker run -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
           -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
           -e AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} \
            --name object_detection_tflite_container \
            --rm -i -t object_detection_tflite
python3 -m object_detection_tflite.test_model
```