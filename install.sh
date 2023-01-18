conda create --name wsss python=3.8 -y 

conda activate wsss

conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 -c pytorch -y

pip3 install pandas gpustat matplotlib numpy

conda install git wget -y

sudo apt-get update
sudo apt-get install tmux

conda install tensorboardX -y

pip install pyyaml imageio

cd data
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar