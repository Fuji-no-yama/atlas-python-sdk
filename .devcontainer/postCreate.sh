#!/bin/bash
set -e

# PATHの設定 (claude codeのインストール先)
CLAUDE_BIN_DIR="/root/.local/bin"
if ! grep -q "$CLAUDE_BIN_DIR" /root/.bashrc; then
    echo "export PATH=\"$CLAUDE_BIN_DIR:\$PATH\"" >> /root/.bashrc
fi
export PATH="$CLAUDE_BIN_DIR:$PATH"

# Python仮想環境のセットアップ
rm -rf /workspace/.venv
uv sync

# claude.jsonの設定 (ホストの認証情報を引き継ぎ、installMethodをコンテナに合わせて上書き)
python3 - <<'EOF'
import json

host_config = "/root/.claude.json.host"
content = open(host_config).read()
d = json.loads(content) if content.strip() else {}
d["installMethod"] = "native"
json.dump(d, open("/root/.claude.json", "w"))
EOF
