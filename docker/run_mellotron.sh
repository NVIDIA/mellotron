#!/bin/bash

if [ 1 -eq $# ]
then
   imageid=$1
   export GID=$(id -g)
   sudo docker container run --rm -it --gpus all \
       	-v ${HOME}/projects/mellotron:/workspace/mellotron \
      	-v ${HOME}/projects/LibriTTS/train-clean-100:/path_to_libritts \
      	-v ${HOME}/projects/LJSpeech-1.1/wavs:/path_to_ljs \
      	-p 9876:8888 -p 9875:6006 \
	      ${imageid} /workspace/add_user.sh ${USER} ${UID}
else
  echo "usage: ${0} <image name>"
fi
