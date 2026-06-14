#!/bin/bash
set -e

sudo apt install python3.14-venv -y

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

exit 0