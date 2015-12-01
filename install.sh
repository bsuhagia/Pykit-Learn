#!/bin/sh

# Install pip if not already installed
if ! pip_loc="$(type -p "pip")" || [ -z "$pip_loc" ]; then
    sudo curl https://bootstrap.pypa.io/ez_setup.py -o - | sudo python
fi
sudo easy_install pip

# Python module dependencies
sudo pip install sklearn
sudo pip install numpy
sudo pip install matplotlib
sudo pip install pandas
sudo pip install scipy
sudo pip install seaborn
sudo pip install Pillow
sudo pip install nose

PROJ_DIR=`pwd`
echo `export PYTHONPATH=$PYTHONPATH:$PROJ_DIR` >> ~/.bash_profile

