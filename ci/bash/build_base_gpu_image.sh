docker build --build-arg CUDA_VERSION=$CUDA_VERSION -t base-gpu-${CUDA_VERSION} -f Dockerfile.base.gpu .

DOCKER_REGISTRY="ghcr.io/weaviate/t2v-transformers-models-rs"

docker tag base-gpu-${CUDA_VERSION} $DOCKER_REGISTRY/base-gpu-${CUDA_VERSION}:latest

docker push $DOCKER_REGISTRY/base-gpu-${CUDA_VERSION}:latest

docker logout