#!/bin/bash

if ! python3 -V &> /dev/null; then
    echo "未找到 Python，终止运行"
    read -p "按 Enter 键继续..."
    exit 1
fi

if ! python3 -c "import venv" &> /dev/null; then
    echo "未找到 Python venv 模块，终止运行"
    read -p "按 Enter 键继续..."
    exit 1
fi

cd "$(cd "$(dirname "$0")" ; pwd)"

if [[ ! -f "./venv/bin/activate" ]]; then
    echo "创建虚拟环境中..."
    python3 -m venv venv
fi

source ./venv/bin/activate

python3 launch.py "$@"

read