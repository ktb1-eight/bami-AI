#!/bin/bash

# 1. Docker 로그인 (ECR)
aws ecr get-login-password --region ap-northeast-2 | docker login --username AWS --password-stdin $ECR_URI

# 2. 최신 Docker 이미지 가져오기
echo "Pulling the latest Docker image..."
docker pull $ECR_URI:latest

# 3. 기존 컨테이너 중지 및 삭제
echo "Stopping and removing old containers..."
docker stop bami-long || true
docker rm bami-long || true

# 4. 새로운 컨테이너 실행 (이름: bami-long)
echo "Running the new Docker container..."
docker run -d \
  --name bami-long \
  -p 8000:8000 \
  $ECR_URI:latest

# 5. 배포 완료 메시지
echo "Deployment completed successfully!"
