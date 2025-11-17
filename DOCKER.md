# Docker Setup Guide

## Quick Start

### Build and run with Docker Compose (recommended)
```bash
docker-compose up -d
```

### Or build and run manually
```bash
# Build the image
docker build -t mammogram-api .

# Run the container
docker run -d -p 8000:8000 --name mammogram-api mammogram-api
```

## Access the API

- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## Docker Commands

### View logs
```bash
docker-compose logs -f
```

### Stop the service
```bash
docker-compose down
```

### Rebuild after changes
```bash
docker-compose up -d --build
```

### Check container status
```bash
docker-compose ps
```

## Testing the Dockerized API

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test3.png"
```

## Troubleshooting

### View container logs
```bash
docker logs mammogram-api
```

### Access container shell
```bash
docker exec -it mammogram-api bash
```

### Check resource usage
```bash
docker stats mammogram-api
```
