# Online install

```
pip install git+ssh://git@github.com/Intelligence-Design/ml_function.git@feature/first#subdirectory=object_attribute/trt
```

# Docker image build
## For jetson_xavier_nx_jp461
```
sudo DOCKER_BUILDKIT=1 docker build --secret id=ssh,src=${HOME}/.ssh/id_rsa -f jetson_xavier_nx_jp461.Dockerfile -t object_attribute_trt .
```

## For g4dn_xlarge_Deep_Learning_Base_AMI_Ubuntu_18_04_Version_44_0
```
sudo DOCKER_BUILDKIT=1 docker build --secret id=ssh,src=${HOME}/.ssh/id_rsa -f g4dn_xlarge_Deep_Learning_Base_AMI_Ubuntu_18_04_Version_44_0.Dockerfile -t object_attribute_trt .
```


# Docker run
##For jetson_xavier_nx_jp461 Docker container run

```
sudo docker run --runtime nvidia \
           -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
           -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
           -e AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} \
            --name object_attribute_trt_container \
            --rm -i -t object_attribute_trt
python3 -m object_attribute_trt.test_model_jetson_xavier_nx
```

##For jetson_xavier_nx_jp461 Docker container run

```
sudo docker run --runtime nvidia \
           -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
           -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
           -e AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} \
            --name object_attribute_trt_container \
            --rm -i -t object_attribute_trt
python3 -m object_attribute_trt.test_model_g4dn_xlarge
```