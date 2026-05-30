#!/bin/bash
# Build and run the OWUI tool-server container.
# Joins the same docker network as the open-webui container so OWUI can
# reach it by container name (http://owui-tool-server:8001).
set -euo pipefail

TOOL_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="owui-tool-server:latest"
CONTAINER="owui-tool-server"
BACKUP_IMAGE="owui-tool-server:rollback"
HOST_PORT=8001
ENV_FILE="${TOOL_DIR}/tool-server.env"
ENV_ARGS=()
if [ -f "$ENV_FILE" ]; then
    ENV_ARGS=(--env-file "$ENV_FILE")
fi

# Discover the network the existing open-webui container lives on.
NETWORK=$(docker inspect open-webui \
    --format '{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{"\n"}}{{end}}' \
    2>/dev/null | head -n 1)
if [ -z "$NETWORK" ]; then
    echo "ERROR: cannot find open-webui container or its network. Is OWUI running?"
    exit 1
fi
echo "Joining network: $NETWORK"

if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "--- Saving rollback image ---"
    docker tag "$IMAGE" "$BACKUP_IMAGE"
fi

echo "--- Building image ---"
docker build -t "$IMAGE" "$TOOL_DIR"

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "--- Stopping + removing old container ---"
    docker stop "$CONTAINER" >/dev/null
    docker rm "$CONTAINER" >/dev/null
fi

echo "--- Running new container ---"
if ! docker run -d \
        --name "$CONTAINER" \
        --network "$NETWORK" \
        --restart unless-stopped \
        "${ENV_ARGS[@]}" \
        -p "127.0.0.1:${HOST_PORT}:8001" \
        "$IMAGE"; then
    echo "ERROR: failed to start new tool-server container"
    if docker image inspect "$BACKUP_IMAGE" >/dev/null 2>&1; then
        docker run -d \
            --name "$CONTAINER" \
            --network "$NETWORK" \
            --restart unless-stopped \
            "${ENV_ARGS[@]}" \
            -p "127.0.0.1:${HOST_PORT}:8001" \
            "$BACKUP_IMAGE" >/dev/null
    fi
    exit 1
fi

echo "--- Waiting for /health (max 60s) ---"
for i in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:${HOST_PORT}/health" >/dev/null 2>&1; then
        echo "OK after ${i} attempt(s)"
        curl -s "http://127.0.0.1:${HOST_PORT}/health"
        echo
        echo
        echo "Reachable from OWUI as: http://${CONTAINER}:8001"
        exit 0
    fi
    sleep 2
done

echo "FAILED to become healthy. Last logs:"
docker logs --tail 50 "$CONTAINER"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
if docker image inspect "$BACKUP_IMAGE" >/dev/null 2>&1; then
    echo "--- Rolling back to previous tool-server image ---"
    docker run -d \
        --name "$CONTAINER" \
        --network "$NETWORK" \
        --restart unless-stopped \
        "${ENV_ARGS[@]}" \
        -p "127.0.0.1:${HOST_PORT}:8001" \
        "$BACKUP_IMAGE" >/dev/null
fi
exit 1
