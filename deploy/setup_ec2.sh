#!/usr/bin/env bash
# One-shot setup on a FRESH Ubuntu 22.04 / 24.04 EC2 instance.
# Run from inside the unzipped folder:  bash deploy/setup_ec2.sh
set -e
APP_DIR=/home/ubuntu/air-quality-aws
cd "$(dirname "$0")/.."
if [ "$(pwd)" != "$APP_DIR" ]; then
  echo "Copying project to $APP_DIR"; mkdir -p "$APP_DIR"; cp -r . "$APP_DIR"; cd "$APP_DIR"
fi

sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip sqlite3

# 2 GB swap so TensorFlow training does not run out of memory on small instances
if ! swapon --show | grep -q swapfile; then
  sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
MEM_MB=$(free -m | awk '/Mem:/{print $2}')
if [ "$MEM_MB" -ge 1800 ]; then
  ./venv/bin/pip install -r requirements-dl.txt || echo "TensorFlow install failed - LSTM/GRU will be disabled"
else
  echo "Less than 2 GB RAM: skipping TensorFlow (LSTM/GRU disabled). Use t3.small or larger for all 5 models."
fi

# (re)train the models with THIS server's library versions
if ./venv/bin/python -c "import tensorflow" 2>/dev/null; then
  ./venv/bin/python train.py
else
  ./venv/bin/python train.py --no-dl
fi

sudo cp deploy/aqms.service deploy/aqms-retrain.service deploy/aqms-retrain.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now aqms
sudo systemctl enable --now aqms-retrain.timer
sleep 5
curl -s http://127.0.0.1:5000/api/health && echo
echo "Done. Open http://<EC2_PUBLIC_IP>:5000  (security group must allow TCP 5000)"
