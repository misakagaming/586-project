curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install -y git-lfs
git lfs install
git clone https://huggingface.co/microsoft/graphcodebert-base
