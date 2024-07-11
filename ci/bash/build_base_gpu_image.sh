DOCKER_REGISTRY="ghcr.io/weaviate/t2v-transformers-models-rs"
IMAGE=$DOCKER_REGISTRY/base-gpu-${CUDA_VERSION}:latest

docker build --cache-from $IMAGE --build-arg CUDA_VERSION=$CUDA_VERSION -t base-gpu -f Dockerfile.base.gpu .

docker tag base-gpu $IMAGE

docker push $IMAGE

docker logout