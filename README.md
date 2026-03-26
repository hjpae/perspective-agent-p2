# perspective-agent-p2

## Vast.ai instance setup 

git config --global user.name "hjpae"  
git config --global user.email "hnjpae@gmail.com"  

ssh-keygen -t ed25519 -C "hnjpae@gmail.com"  
eval "$(ssh-agent -s)"  
ssh-add ~/.ssh/id_ed25519  
cat ~/.ssh/id_ed25519.pub  

git clone git@github.com:hjpae/perspective-agent-p2.git

ssh -T git@github.com  
cd perspective-agent-p2  
git remote -v  

git remote remove origin  
git remote add origin git@github.com:hjpae/perspective-agent-p2.git  

conda env create -f environment.yml  
conda activate cear-phase2  


## install dependencies (cuda 12.8 ... code is 13.1 compatible)  

conda env create -f environment.yml  
conda activate cear-phase2  

pip install --no-cache-dir torch torchvision torchaudio \  
  --index-url https://download.pytorch.org/whl/cu128  

python - <<'PY'  
import torch  
print(torch.__version__)  
print(torch.version.cuda)  
print(torch.cuda.is_available())  
if torch.cuda.is_available():  
    print(torch.cuda.get_device_name(0))  
PY  