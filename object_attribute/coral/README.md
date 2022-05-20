# Online install

```
pip install git+ssh://git@github.com/Intelligence-Design/ml_function.git@feature/first#subdirectory=object_attribute/coral
```

# Docker image build

```
DOCKER_BUILDKIT=1 docker build --secret id=ssh,src=${HOME}/.ssh/id_rsa -t object_attribute_coral .
```

# Docker container run

```
docker run -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
           -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
           -e AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} \
            --name object_attribute_coral_container \
            --rm -i -t  --privileged -v /dev/bus/usb:/dev/bus/usb object_attribute_coral
python3 -m object_attribute_tflite.test_model
```