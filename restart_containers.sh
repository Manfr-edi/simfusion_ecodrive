pip install docker
xhost +local:docker
#!/bin/bash

CONTAINERS="opensbt_run carla_sim autoware_mini_lofi autoware_mini_hifi" 

# Corresponding service names
declare -A SERVICE_MAP=(
    [autoware_mini_lofi]="autoware_mini_lofi"
    [autoware_mini_hifi]="autoware_mini_hifi"
    [opensbt]="opensbt_run"
    [carla_sim]="carla_sim"
)

echo "Deleting containers..."
for container in $CONTAINERS; do
    docker stop "$container" 2>/dev/null || true
    docker rm "$container" 2>/dev/null || true
    echo "Deleted: $container"
done

echo "Waiting 3 seconds..."
sleep 3

echo "Recreating containers..."
for container in $CONTAINERS; do
    SERVICE_NAME="${SERVICE_MAP[$container]}"
    echo "Starting: $SERVICE_NAME"
    docker compose up -d "$SERVICE_NAME"
done

echo "Done!"
