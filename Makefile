


IMAGE_NAME = ml-model-deployment


DOCKER_USERNAME = samansaeed2306

require:
	pip install -r requirements.txt

skicit:
    pip install scikit-learn joblib


build:
    docker build -t $(IMAGE_NAME) .


run:
    docker run -p 5000:5000 $(IMAGE_NAME)


push:
    docker tag $(IMAGE_NAME) $(DOCKER_USERNAME)/$(IMAGE_NAME)
    docker push $(DOCKER_USERNAME)/$(IMAGE_NAME)
