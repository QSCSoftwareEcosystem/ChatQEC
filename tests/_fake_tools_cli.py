"""Stand-in for ``chatqec-tools`` used by test_cli_tools_client.py."""

import json
import os
import sys
import time

TOOLS = [
    {"name": "echo", "description": "returns its args", "input_schema": {"type": "object"}},
    {"name": "slow", "description": "sleeps", "input_schema": {}},
]

if sys.argv[1] == "list":
    print(json.dumps(TOOLS))
    sys.exit(0)

name = sys.argv[2]
args = json.loads(sys.stdin.read() or "{}")
if name == "echo":
    print(json.dumps({"ok": True, "args": args, "env": os.environ.get("FAKE_TOOLS_ENV")}))
elif name == "slow":
    time.sleep(30)
elif name == "garbage":
    print("not json")
elif name == "crash":
    print("boom", file=sys.stderr)
    sys.exit(3)
