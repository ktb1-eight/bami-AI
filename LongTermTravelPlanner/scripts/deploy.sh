#!/bin/bash

# 1. Docker 로그인 (ECR)
echo "Logging into Docker ECR..."
aws ecr get-login-password --region ap-northeast-2 | docker login --username AWS --password-stdin $ECR_URI
if [ $? -ne 0 ]; then
    echo "Docker login failed"
    exit 1
else
    echo "Docker login successful"
fi

# 2. 최신 Docker 이미지 가져오기
echo "Pulling the latest Docker image..."
docker pull $ECR_URI:latest
if [ $? -ne 0 ]; then
    echo "Failed to pull Docker image"
    exit 1
else
    echo "Docker image pulled successfully"
fi

# 3. 기존 컨테이너 중지 및 삭제
echo "Stopping and removing old containers..."
docker stop bami-long || true
docker rm bami-long || true
if [ $? -ne 0 ]; then
    echo "Failed to stop or remove old containers"
    exit 1
else
    echo "Old containers stopped and removed"
fi

# 4. 새로운 컨테이너 실행 (이름: bami-long)
echo "Running the new Docker container..."
docker run -d \
  --name bami-long \
  -p 8000:8000 \
  $ECR_URI:latest
if [ $? -ne 0 ]; then
    echo "Failed to run Docker container"
    exit 1
else
    echo "New Docker container is running"
fi

# 5. 배포 완료 메시지
echo "Deployment completed successfully!"
