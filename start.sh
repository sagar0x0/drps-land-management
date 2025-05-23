#!/bin/bash
version="v2.0.0"
current_dir="$(cd "$(dirname "$0")" && pwd)"
network_name="docker_net"

docker pull xinfinorg/subnet-generator:$version
mkdir -p generated/scripts

if ! docker network inspect "$network_name" > /dev/null 2>&1; then
  echo "Network '$network_name' does not exist. Creating it..."
  docker network create --subnet 192.168.25.0/24 "$network_name"
else
  echo "Joining existing network '$network_name'"
fi

docker run -d                                   \
  --network "docker_net" --ip=192.168.25.111    \
  -p 5210:5210                                  \
  -v /var/run/docker.sock:/var/run/docker.sock  \
  -v $current_dir/generated:/mount/generated    \
  -e HOSTPWD=$current_dir/generated             \
  xinfinorg/subnet-generator:$version           \
  && \
echo '' && \
echo '' && \
echo '' && \
echo 'if this is running on your server, first use ssh tunnel: ssh -N -L localhost:5210:localhost:5210 <username>@<ip_address> -i <private_key_file>' && \
echo 'if you are using VSCode Remote Explorer, ssh tunnel will be available by default' && \
echo 'http://localhost:5210 to access Subnet Deployment Wizard'
