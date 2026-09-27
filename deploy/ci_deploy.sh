#!/usr/bin/env bash
# Runs ON the EC2 after GitHub Actions has rsynced the new code into /home/ubuntu/air-quality-aws.
# Installs changed requirements, refreshes the systemd unit and restarts the app.
# First-time setup is deploy/setup_ec2.sh (creates venv, swap, trains models).
set -euo pipefail
APP_DIR=/home/ubuntu/air-quality-aws
cd "$APP_DIR"

if [ ! -x venv/bin/python ]; then
  echo "No venv yet - running full first-time setup"
  exec bash deploy/setup_ec2.sh
fi

# reinstall requirements only when they changed
for f in requirements.txt requirements-dl.txt; do
  sum=$(sha256sum "$f" | cut -d' ' -f1)
  if [ "$(cat ".$f.sha" 2>/dev/null)" != "$sum" ]; then
    if [ "$f" = requirements.txt ] || venv/bin/python -c "import tensorflow" 2>/dev/null; then
      echo "Installing $f"; venv/bin/pip install -q -r "$f"
    fi
    echo "$sum" > ".$f.sha"
  fi
done

# models are trained on the server; train once if they are missing
[ -f trained_models/training_metrics.json ] || venv/bin/python train.py --quick

if ! cmp -s deploy/aqms.service /etc/systemd/system/aqms.service; then
  sudo cp deploy/aqms.service /etc/systemd/system/aqms.service
  sudo systemctl daemon-reload
fi
sudo systemctl enable aqms >/dev/null 2>&1
sudo systemctl restart aqms

for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:5000/api/health >/dev/null; then
    echo "Health OK:"; curl -s http://127.0.0.1:5000/api/health; echo; exit 0
  fi
  sleep 2
done
echo "Health check FAILED"; sudo journalctl -u aqms -n 50 --no-pager; exit 1
