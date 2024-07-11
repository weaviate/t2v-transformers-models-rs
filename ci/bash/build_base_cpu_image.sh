docker build -t base-cpu -f Dockerfile.base.cpu .

DOCKER_REGISTRY="ghcr.io/weaviate/t2v-transformers-models-rs"

docker tag base-cpu $DOCKER_REGISTRY/base-cpu:latest

docker push $DOCKER_REGISTRY/base-cpu:latest

docker logout