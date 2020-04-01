# This is a Dockerfile and runscript to help you get going quickly
- It makes container run as you instad of root
- Automatically starts Tensorboard
## It makes the following assumptions:
* You are running Ubuntu
* You have a projects folder under your Home folder
* Under your projects folder you have cloned Mellotron
* Also under your projects folder you have un-tarred the LibriTTS and or LJSpeech
## Running the script:
To run the container provide the tag you gave your image when you built it.
...
pneumoman@zombunnie:~/projects/mellotron$  ./run_mellotron.sh mellotronix
...
## Running Training
Use full paths! I.E.
...
pneumoman@2a57e3b56769:/workspace/mellotron$ python train.py --output_directory=/workspace/mellotron/models --log_directory=/workspace/logs
...
