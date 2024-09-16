"""
Utility functions for operationan code
"""

import json
import os
import platform
import sys
import time
from pathlib import Path
from subprocess import CompletedProcess, run
from types import ModuleType
from typing import Any

from ulid import ULID
from rich import pretty
pretty.install()
from snoop import pp

class Ansible:
    """
    Ansible context
    """

    _results: list | None = []
    vars: dict[str, Any] = {}
    hostlist: list[str] = []
    hosts: dict[str, dict[str, Any]] = {}
    hostgroups: dict[str, Any] = {}
    host: str

    def __init__(self) -> None:
        self.__dict__.update(pp(json.load(sys.stdin)))

    def __enter__(self):
        return self

    def __exit__(self):
        """
        Print results it there are any
        """
        self.results()

    def __del__(self):
        """
        Prints results if there are any
        """
        self.results()

    def run(self, module: ModuleType, **parameters: Any) -> dict[str, Any]:
        """
        Runs an Ansible module through a python command

        Args:
            module (ModuleType): Ansible module
            **parameters (dict[str, Any]): Module parameters

        Returns:
            dict[str, Any]: Result, consisting of exit status from command, stdout and stderr (see CompleteledProcess)
        """
        while True:
            id = str(ULID())
            outputDir = Path(os.environ.get("_MANAGE_OUTPUT_PATH", '.')) / id
            try:
                outputDir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                pass
            else:
                break
        skipped = parameters.pop('_skip', False)
        input_ = json.dumps(
            dict(
                ANSIBLE_MODULE_ARGS=(
                    parameters
                    | dict(_ansible_check_mode=os.environ['_MANAGE_CHECK_MODE'], _ansible_diff=os.environ['_MANAGE_DIFF_MODE'])
                )
            ),
            separators=(',', ':'),
        )
        with (outputDir / 'parameters.json').open('w') as parametersStream:
            json.dump(
                dict(
                    module=module.__name__,
                    parameters=parameters,
                    vars=self.vars,
                    hostlist=self.hostlist,
                    hosts=self.hosts,
                    hostgroups=self.hostgroups,
                    host=platform.node(),
                ),
                parametersStream,
                indent=2,
            )
        time_ = 0.0
        with (outputDir / "stdout").open(mode='w') as stdout, (outputDir / "stdout").open(mode='w') as stderr:
            if skipped:
                execution = CompletedProcess(args=['python3', '-m', module.__name__], returncode=0, stdout='', stderr='')
            else:
                time_ = time.time()
                execution = run(
                    ['python3', '-m', module.__name__],
                    input=input_,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    cwd=os.path.dirname(__file__),
                )
                time_ = time.time() - time_
        ansibleResult: dict[str, Any]
        with (outputDir / 'result').open('w') as resultStream:
            result = (outputDir / 'stdout').read_text()
            try:
                ansibleResult = json.loads(result)
            except json.JSONDecodeError:
                ansibleResult = dict(
                    failed=execution.returncode != 0,
                    changed=False,
                    msg=result[:1024] + '…' if len(result) > 1024 else '',
                )
            ansibleResult.update(skipped=skipped)
            ansibleResult = dict(
                status=execution.returncode, ansible_result=ansibleResult, vars=self.vars, host=platform.node(), runtime=time_
            )
            ansibleJson = json.dumps(ansibleResult)
            resultStream.write(ansibleJson)
            if self._results is None:
                self._results = []
            self._results.append(ansibleJson)
        return ansibleResult

    def results(self):
        """
        Print results if there are any
        """
        if self._results is not None:
            print('[')
            comma = ''
            for result in self._results:
                print(comma, '  ', result, end='', sep='')
                comma = ',\n'
            print(']')
            self._results = None
