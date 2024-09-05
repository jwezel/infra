import json
from subprocess import run
from types import ModuleType
from typing import Any
import os


def Ansible(module: ModuleType, parameters: Any) -> int:
    return run(
        ['python3', '-m', module.__name__],
        input=json.dumps(dict(ANSIBLE_MODULE_ARGS=parameters)),
        text=True,
        cwd=os.path.dirname(__file__),
    ).returncode
