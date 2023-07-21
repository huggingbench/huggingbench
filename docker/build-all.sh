# Here we build docker images that are used by the exporter to generate different model formats and
# optimize the models. We currently support GPU, Intel CPU and Apple M1/M2 architectures.
#!/bin/bash

SCRIPT_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

if [[ $(uname -m) == 'arm64' ]]; then
  MAC=true
else
    MAC=false
fi


if [ "$MAC" = false ] ; then

    while true; do

    read -p "Do you have a dedicated GPU with NVidia CUDA drivers? (y/n)" yn

    case $yn in 
        [yY] ) GPU=true;
            break;;
        [nN] ) GPU=false;
            break;;
        * ) echo invalid response;;
    esac

    done

else
    GPU=false
fi


if [ "$GPU" = true ] ; then
    echo "Building GPU Huggingface Optimum image"
    docker build --network=host -t optimum -f $SCRIPT_PATH/optimum/Dockerfile.gpu $SCRIPT_PATH/optimum/
    echo "Building OpenVINO image"
    docker build --network=host -t openvino $SCRIPT_PATH/openvino/
elif [ "$MAC" = false ] ; then
    echo "Building CPU Huggingface Optimum image"
    docker build --network=host -t optimum -f $SCRIPT_PATH/optimum/Dockerfile.cpu $SCRIPT_PATH/optimum/
    echo "Building OpenVINO image"
    docker build --network=host -t openvino $SCRIPT_PATH/openvino/   
else
    echo "Building Apple M1/M2 Huggingface Optimum image"
    docker build --network=host -t optimum -f $SCRIPT_PATH/optimum/Dockerfile.cpu.arm64 $SCRIPT_PATH/optimum/
    echo "Can't build OpenVINO image on Apple M1/M2 architecture" 
fi

echo "Building Polygraphy image"
docker build --network=host -t polygraphy $SCRIPT_PATH/polygraphy/
