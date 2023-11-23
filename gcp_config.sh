sudo apt update
sudo apt autoremove -y
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.10 -y
# install cuda version 11.7
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
sudo apt -y install nvidia-cuda-toolkit
# prepare torch environment
sudo apt install -y python3-pip
sudo pip3 install torch torchvision torchaudio
# install venv
sudo apt install python3.10-venv

# reference
# https://koding.work/how-to-install-cuda-and-cudnn-to-ubuntu-20-04/#Step_4_cuDNN
# sudo apt install nvidia-cuda-toolkit
# https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local
# https://developer.nvidia.com/rdp/cudnn-archive (according to CUDA version shown in nvidia-smi )
# tar -Jxvf cudnn-linux-x86_64-8.9.5.30_cuda12-archive.tar.xz
# sudo cp cuda/include/cudnn*.h /usr/local/cuda/include 
# sudo cp -P cuda/lib/libcudnn* /usr/local/cuda/lib64
# sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
