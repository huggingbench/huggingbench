# Here we build docker images that are used by the exporter to generate different model formats and
# optimize the models
#!/bin/bash

SCRIPT_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

while true; do

read -p "Do you have a dedicated GPU/cuda? (y/n)" yn

case $yn in 
	[yY] ) GPU=true;
		break;;
	[nN] ) GPU=false;
		break;;
	* ) echo invalid response;;
esac

done

if [ "$GPU" = true ] ; then
    echo "Building GPU Huggingface Optimum image"
    docker build -t optimum -f $SCRIPT_PATH/optimum/Dockerfile.gpu $SCRIPT_PATH/optimum/
else
    echo "Building CPU Huggingface Optimum image"
    docker build -t optimum -f $SCRIPT_PATH/optimum/Dockerfile.cpu $SCRIPT_PATH/optimum/
fi

docker build -t openvino $SCRIPT_PATH/openvino/
docker build -t polygraphy $SCRIPT_PATH/polygraphy/
