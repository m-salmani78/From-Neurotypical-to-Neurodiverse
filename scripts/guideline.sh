### Linux CUDA ###

gcc --version
### Check wich cuda version installed (Version 12.2 is recommended)
ls /usr/local/ | grep cuda

### If it was any version else
sudo rm -r /usr/local/cuda-12.1/
sudo apt clean && sudo apt autoclean
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run
# [During CUDA installation]: recommended to cancel the driver installation.

# Add CUDA 12.2 to PATH
export PATH=/usr/local/cuda-12.2/bin:$PATH
# Add CUDA 12.2 to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH

### After completion, enter to check whether the corresponding version number appears.
nvcc -V

### Installing VLLM ###
uv pip install vllm --torch-backend=auto
rm cuda_12.2.0_535.54.03_linux.run

### Installing LLaMA-Factory ###
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
# [*] If there is an environment conflict, try resolving it using `pip install --no-deps -e`
pip install -e ".[torch,metrics]"
llamafactory-cli version

# [*] ./LLaMA-Factory/src/llamafactory/data/converter.py

### Clone The Project ###
git clone https://x-access-token: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@github.com/m-salmani78/Minds-in-the-Machine
# [*] Put the config.json file in `./Minds-in-the-Machine` path
chmod +x ./Minds-in-the-Machine/src/actor-critic-loop/run_evaluation.sh

# Run the workflow (e.g. level=1 , actor-critic , gemma-2-b-it)
./Minds-in-the-Machine/src/actor-critic-loop/run_evaluation.sh --model_name google/gemma-2-2b-it --template gemma --level 1

git config --global user.name "Your Name"
git config --global user.email "you@example.com"
