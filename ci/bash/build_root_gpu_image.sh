docker build --build-arg CUDA_VERSION=$CUDA_VERSION -t t2v-rs-root-gpu-${CUDA_VERSION} -f Dockerfile.root.gpu .

DOCKER_REGISTRY="ghcr.io/weaviate/t2v-transformers-models-rs"

docker tag t2v-rs-root-gpu-${CUDA_VERSION} $DOCKER_REGISTRY/t2v-rs-root-gpu-${CUDA_VERSION}:0.0.1

docker push $DOCKER_REGISTRY/t2v-rs-root-gpu-${CUDA_VERSION}:0.0.1

docker logout