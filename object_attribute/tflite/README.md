# Online install

```
pip install git+ssh://git@github.com/Intelligence-Design/ml_function.git@feature/first#subdirectory=object_attribute/tflite
```

# Docker image build

```
DOCKER_BUILDKIT=1 docker build --secret id=ssh,src=${HOME}/.ssh/id_rsa -t object_attribute_tflite .
```

# Docker container run

```
docker run --name object_attribute_tflite_container --rm -i -t object_attribute_tflite
```