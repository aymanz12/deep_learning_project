#!/bin/bash

# Script to build and push Docker image to DockerHub
# Usage: ./build_and_push.sh <your-dockerhub-username>

if [ -z "$1" ]; then
    echo "Usage: ./build_and_push.sh <your-dockerhub-username>"
    echo "Example: ./build_and_push.sh myusername"
    exit 1
fi

DOCKERHUB_USERNAME=$1
IMAGE_NAME="defect-detection"
TAG="latest"

echo "🔨 Building Docker image..."
docker build -t ${IMAGE_NAME}:${TAG} .

echo "🏷️  Tagging image for DockerHub..."
docker tag ${IMAGE_NAME}:${TAG} ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG}

echo "📤 Pushing to DockerHub..."
docker push ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG}

echo "✅ Done! Image pushed to: ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG}"
echo ""
echo "To pull and run:"
echo "  docker pull ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG}"
echo "  docker run -d --name defect-detection -p 7860:7860 ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG}"






