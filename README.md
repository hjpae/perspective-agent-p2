# perspective-agent-p2

## Vast.ai instance setup 

git config --global user.name "hjpae"
git config --global user.email "hjpae@gmail.com"

ssh-keygen -t ed25519 -C "hjpae@gmail.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub

ssh -T git@github.com
cd perspective-agent-p2
git remote -v

git remote remove origin
git remote add origin git@github.com:hjpae/perspective-agent-p2.git

conda env create -f environment.yml
conda activate cear-phase2