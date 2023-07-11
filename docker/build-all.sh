# Here we build docker images that are used by the exporter to generate different model formats and
# optimize the models
#!/bin/bash

SCRIPT_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

#!/bin/bash

# Just a bad demonstration
check_intel_cpu() {
  if [[ "$(uname)" == "Darwin" ]]; then
    machine_cpu=$(sysctl -n machdep.cpu.brand_string)
  else
    machine_cpu=$(grep -i 'model name' /proc/cpuinfo | head -n 1 | cut -d ':' -f 2 | sed 's/^[[:space:]]*//')
  fi

  if echo "$machine_cpu" | grep -qi 'GenuineIntel'; then
    echo "Building Dockerfile for Intel CPUs in openvino/ directory..."
    docker build -t openvino $SCRIPT_PATH/openvino/
  else
    echo "Error: Machine does not have Intel CPUs. Unable to build Dockerfile.
  fi
}

# Function to check if the machine has NVIDIA GPUs
check_nvidia_gpu() {
  if [[ "$(uname)" == "Darwin" ]]; then
    machine_gpu=$(system_profiler SPDisplaysDataType 2>/dev/null | grep -i 'NVIDIA')
  else
    machine_gpu=$(lspci | grep -i 'NVIDIA')
  fi

  if [[ -n "$machine_gpu" ]]; then
    echo "Building Dockerfile for NVIDIA GPUs in polygraphy/ directory..."
    docker build -t optimum -f $SCRIPT_PATH/optimum/Dockerfile.gpu $SCRIPT_PATH/optimum/
  else
    docker build -t optimum -f $SCRIPT_PATH/optimum/Dockerfile.cpu $SCRIPT_PATH/optimum/
  fi
}


while true; do

read -p "Do you have a dedicated GPU? (y/n)" yn

case $yn in 
	[yY] ) GPU=true;
		break;;
	[nN] ) GPU=false;
		break;;
	* ) echo invalid response;;
esac

done

if [ "$GPU" = true ] ; then
    check_nvidia_gpu
    echo "Building GPU Huggingface Optimum image"
    docker build -t optimum -f $SCRIPT_PATH/optimum/Dockerfile.gpu $SCRIPT_PATH/optimum/
else
    echo "Building CPU Huggingface Optimum image"
    docker build -t optimum -f $SCRIPT_PATH/optimum/Dockerfile.cpu $SCRIPT_PATH/optimum/
fi

docker build -t polygraphy $SCRIPT_PATH/polygraphy/
