#!/bin/bash
# retry wrapper: the box is shared and python processes occasionally get killed; checkpoints make reruns cheap
cd "$(dirname "$0")"
for i in 1 2 3 4 5 6; do
  ../../.venv/Scripts/python.exe -u run.py "$@" && exit 0
  echo "run failed/killed (attempt $i), retrying" >&2
  sleep 5
done
exit 1
