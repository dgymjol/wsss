conda create --name wsss python=3.8 -y 

conda activate wsss

conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 -c pytorch -y
# pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html

pip install pandas gpustat matplotlib numpy

conda install git wget -y

conda install tensorboardX -y

pip install pyyaml imageio opencv-python

sudo apt-get update
sudo apt-get install tmux
# sudo apt-get install libgl1-mesa-glx
# sudo apt-get install libglib2.0-0

conda install -c conda-forge pydensecrf

cd data
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar


# sudo apt-get update
# sudo apt-get install build-essential libssl-dev libffi-dev python-dev
pip install chainercv