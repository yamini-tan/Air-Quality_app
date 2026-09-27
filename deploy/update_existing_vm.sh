#!/usr/bin/env bash
# Upgrade a RUNNING VM that has the old version in /home/ubuntu/air-quality-aws.
# Upload the new zip to /home/ubuntu first, then:  bash update_existing_vm.sh
set -e
OLD=/home/ubuntu/air-quality-aws
STAMP=$(date +%Y%m%d_%H%M%S)

# 1. stop whatever runs the old app (systemd, screen, nohup ...)
sudo systemctl stop aqms 2>/dev/null || true
pkill -f "python.*app.py" 2>/dev/null || true

# 2. back up the old folder (database included)
cp -r "$OLD" "/home/ubuntu/air-quality-aws_backup_$STAMP"
echo "Backup: /home/ubuntu/air-quality-aws_backup_$STAMP"

# 3. unpack the new code over it, KEEPING the existing sensor_data.db
cd /home/ubuntu
rm -rf /tmp/aqms_new && mkdir /tmp/aqms_new
unzip -q -o air-quality-aws-v2.zip -d /tmp/aqms_new
rsync -a --exclude 'sensor_data.db' --exclude 'venv' /tmp/aqms_new/air-quality-aws/ "$OLD"/
rm -f "$OLD/templates/prediction.html"

# 4. install / train / start
cd "$OLD"
bash deploy/setup_ec2.sh
