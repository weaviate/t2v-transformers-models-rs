DOCKER_REGISTRY="ghcr.io/weaviate/t2v-transformers-models-rs"
IMAGE=$DOCKER_REGISTRY/base-cpu:latest

docker build --cache-from $IMAGE -t base-cpu -f Dockerfile.base.cpu .

docker tag base-cpu $IMAGE

docker push $IMAGE

docker logout