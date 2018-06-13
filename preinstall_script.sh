#!/bin/bash
# May need to uncomment and update to find current packages
# apt-get update

# Required for demo script! #
pip install scikit-video

# Add your desired packages for each workspace initialization
#          Add here!          #
sudo apt-get install cuda-libraries-9-0
sudo dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb
sudo apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub 
sudo apt-get update
sudo apt-get install cuda-9.0
pip install --upgrade tensorflow-gpu