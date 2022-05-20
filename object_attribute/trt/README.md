# Online install

```
pip install git+ssh://git@github.com/Intelligence-Design/ml_function.git@feature/first#subdirectory=object_attribute/tflite
```

# Docker image build
## For jetson_xavier_nx_jp461
```
DOCKER_BUILDKIT=1 docker build --secret id=ssh,src=${HOME}/.ssh/id_rsa -f jetson_xavier_nx_jp461.Dockerfile -t object_attribute_trt .
```

# Docker container run

```
docker run -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
           -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
           -e AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} \
            --name object_attribute_trt_container \
            --rm -i -t object_attribute_trt
```