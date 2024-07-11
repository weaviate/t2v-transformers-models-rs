docker build --build-arg CUDA_VERSION=$CUDA_VERSION -t base-cpu-${CUDA_VERSION} -f Dockerfile.base.cpu .

DOCKER_REGISTRY="ghcr.io/weaviate/t2v-transformers-models-rs"

docker tag base-cpu-${CUDA_VERSION} $DOCKER_REGISTRY/base-cpu-${CUDA_VERSION}:latest

docker push $DOCKER_REGISTRY/base-cpu-${CUDA_VERSION}:latest

docker logout