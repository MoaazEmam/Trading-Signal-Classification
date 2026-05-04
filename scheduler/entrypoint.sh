#!/bin/bash
# scheduler/entrypoint.sh
#
# Installs a cron job that runs the prediction pipeline at 22:00 UTC (01:00 EET)
# Logs go to /app/logs/scheduler.log.

set -e

mkdir -p /app/logs

# 22:00 UTC = 01:00 EET (UTC+3)
CRON_EXPR="0 22 * * 1-5"
CRON_CMD="cd /app && make predict >> /app/logs/scheduler.log 2>&1"

echo "$CRON_EXPR $CRON_CMD" > /etc/cron.d/predict-job
chmod 0644 /etc/cron.d/predict-job
crontab /etc/cron.d/predict-job

echo "Scheduler started. Prediction will run at 1pm for testing"
echo "Logs: /app/logs/scheduler.log"

# keep the env vars available to cron (cron runs in a clean environment)
printenv | grep -v "no_proxy" >> /etc/environment

# start cron in foreground
cron -f