#!/bin/env python3

"""
Manage
"""

import os
import re
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from itertools import batched
from pathlib import Path
from typing import Any, Callable
from shutil import copy2

import config
import plumbum.machines
import yaml
from inventory import Host, Inventory
from paramiko import AutoAddPolicy
from plumbum import local
from plumbum.machines.paramiko_machine import ParamikoMachine
from rich import print, traceback

traceback.install(show_locals=True)

yaml.representer.SafeRepresenter.add_representer(
    str,
    lambda dumper, data: dumper.represent_scalar(
        'tag:yaml.org,2002:str', data, style=[None, '|'][any(c in data for c in '\n\r\x0c\x1d\x1e\x85\u2028\u2029')]
    ),
)

DESCRIPTION = '''Run state scripts'''
NAME_HELP = '''host name or group name (* prefix forces interpretation as host name, @ prefix as group name)'''
PLAY_HELP = '''paths of plays to run'''
CONFIG_HELP = '''configuration settings in the form of NAME=VALUE. Must be last or separated by -- from NAME arguments.'''
EPILOG = '''
Configuration settings available with -s / --set:

config:     path to configuration file
hosts:      path to host inventory file
hostgroups: path to host groups file
'''
REMOTE_ROOT = '/var/tmp/manage'

try:
    import_module('ansible')
    ANSIBLE_PRESENT = True
except ModuleNotFoundError:
    ANSIBLE_PRESENT = False


def CamelCase(text: str) -> str:
    return ''.join(e.title() for e in re.split(r'\W+', text))


def Fqdn(host: Host) -> str:
    hostdomain = host.get('hostdomain')
    return f"{host.name}{'.' + hostdomain if hostdomain else ''}"


def ansible(object, vars) -> dict[str, Any]:
    """
    Run ansible module

    Args:
        object (TYPE): Parameters
    """
    return dict(status=0, out=f'result of {[n for n in object]}, vars={vars}', err='')


def run(object, vars) -> dict[str, Any]:
    return dict(status=0, out=f'run: {object}', err='ERROR: shit happened')


FUNCTION: dict[str, Callable[..., dict[str, Any]]] = {'ansible': ansible, 'run': run}


def ifHandled(task: dict[str, Any], keywords: dict[str, Any], vars: dict[str, Any]) -> bool:
    return False


def doUntilHandled(task: dict[str, Any], keywords: dict[str, Any], vars: dict[str, Any]) -> bool:
    return False


def doWhileHandled(task: dict[str, Any], keywords: dict[str, Any], vars: dict[str, Any]) -> bool:
    return False


def commandsHandled(task: dict[str, Any], keywords: dict[str, Any], vars: dict[str, Any]) -> bool:
    """
    Handle commands

    Args:
        task (dict[str, Any]): Description
        keywords (dict[str, Any]): Description
        vars (dict[str, Any]): Description
    """
    return False


def runYamlTaskList(object: list[dict[str, Any]], vars: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Summary
    """
    results: list[dict[str, Any]] = []
    for taskItem in object:
        keywords = {}
        task = taskItem.copy()
        for keyword in KEYWORDS:
            if keyword in task:
                keywords[keyword] = task.pop(keyword)
        if commandsHandled(task, keywords, vars):
            continue
        for name, function in FUNCTION.items():
            if name in task:
                results.append(function(task[name], vars))
                break
        else:
            raise RuntimeError(f'Task does not call a known function:\n{yaml.safe_dump(taskItem, sort_keys=False)}')
    return results


def runYaml(object: Any) -> list[dict[str, Any]]:
    """
    Run a YAML play list

    Args:
        object (Any): Play list
    """
    assert isinstance(object, list), (
        "YAML play list must be list (ie list of tasks or list " f"of elements being either dict or list), found {type(object)}"
    )
    print(object)
    vars: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    if all(
        isinstance(doc, dict) or isinstance(doc, list) and all(isinstance(element, dict) for element in doc) for doc in object
    ):
        # List of documents
        for document in object:
            if isinstance(document, dict):
                # Variables
                vars.update(document)
            else:
                results.extend(runYamlTaskList(document, vars))
        return pp(results)
    else:
        raise RuntimeError('Invalid YAML task list format')


class LocalMachine(plumbum.machines.LocalMachine):
    """
    LocalMachine with dummy context manager
    """

    def __init__(self, *unused, **kwargs):
        """
        Summary

        Args:
            *unused: Description
        """

    def __enter__(self):
        return self

    def __exit__(self, *ex):
        return

    def copy(self, source: str | Path, remotePath: str):
        """
        Copy files

        Args:
            source (str | Path): Description
        """
        local.cmd.rsync['--archive', '--delete', '--mkpath', source, remotePath]()


class RemoteMachine(ParamikoMachine):
    """
    Summary
    """

    def __init__(
        self,
        host,
        user=None,
        port=None,
        password=None,
        keyfile=None,
        load_system_host_keys=True,
        missing_host_policy=None,
        encoding="utf8",
        look_for_keys=None,
        connect_timeout=None,
        keep_alive=0,
        gss_auth=False,
        gss_kex=None,
        gss_deleg_creds=None,
        gss_host=None,
        get_pty=False,
        load_system_ssh_config=False,
    ):
        super().__init__(
            host,
            user,
            port,
            password,
            keyfile,
            load_system_host_keys,
            missing_host_policy,
            encoding,
            look_for_keys,
            connect_timeout,
            keep_alive,
            gss_auth,
            gss_kex,
            gss_deleg_creds,
            gss_host,
            get_pty,
            load_system_ssh_config,
        )
        self._keyfile = keyfile

    def copy(self, source: str | Path, remotePath: str):
        """
        Copy files

        Args:
            source (str | Path): Description
        """
        local.cmd.rsync[
            '--archive',
            '--delete',
            '--compress',
            '--compress-choice=lz4',
            '--mkpath',
            f"--rsh=ssh{f' -i{self._keyfile}' if self._keyfile else ''}",
            source,
            f"{self.host}:{remotePath}",
        ]()


CONNECTORS: dict[Any, type[RemoteMachine] | type[LocalMachine]] = {
    None: RemoteMachine,
    '': RemoteMachine,
    'local': LocalMachine,
}


def RunPlayList(host: Host, keyfile: str, plays: list[str]) -> dict[str, list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = {}
    with CONNECTORS[host.get('connection')](Fqdn(host), port=host['sshport'], keyfile=keyfile) as remote:
        for play in plays:
            play, _, script = play.partition(':')
            playPath = Path(play)
            remotePlay = sha256(f"{playPath.absolute()}-{playPath.stat().st_ino}".encode()).hexdigest()
            remotePath = remote.path(REMOTE_ROOT) / remotePlay
            remote.copy(play, str(remotePath))
            remotePath = remotePath / playPath.name
            remoteScript = remotePath
            if script:
                remoteScript = remoteScript / script
            results[str(play)] = [dict(zip(('status', 'out', 'err'), remote[remoteScript].run()))]
    return results


@dataclass
class _Context:
    """
    Context
    """

    _args: Namespace
    _config: config.Config
    _hosts: set[Host]

    def run(self) -> list[dict[str, dict[str, Any]]]:
        """
        Run plays

        Args:
            hosts (set[Host]): hosts to target
        """
        results: list[tuple[Host, Future]] = []
        with ThreadPoolExecutor(max_workers=self._config.get('threads')) as executor:
            for hosts in batched(self._hosts, self._config.batchsize):
                for host in hosts:
                    keyfile = host.vars.get('keyfile')
                    if not keyfile:
                        localuser = os.environ["USER"]
                        keydir = self._config.get('keydir') or f'/home/{localuser}/.ssh'
                        user = host.get('user') or localuser
                        keyfile = f"{keydir}/{user}@{Fqdn(host)}"
                    results.append((host, executor.submit(RunPlayList, host, keyfile, self._args.play)))
        return [{h.name: r.result()} for h, r in zip((r[0] for r in results), as_completed(r[1] for r in results))]


def _Main():
    """
    Main
    """
    argp = ArgumentParser(
        description=DESCRIPTION, allow_abbrev=True, epilog=EPILOG, formatter_class=RawDescriptionHelpFormatter
    )
    argp.add_argument('name', nargs='*', default=['ALL'], metavar='NAME', help=NAME_HELP)
    argp.add_argument('-p', '--play', nargs='*', metavar='PLAY', help=PLAY_HELP)
    argp.add_argument('-s', '--set', dest='config', nargs='*', help=CONFIG_HELP)
    args = argp.parse_args()
    config.Init(args.config)
    inventory = Inventory(config.Hosts, config.HostGroups)
    context = _Context(args, config.Settings, inventory.hosts(args.name))
    # print([(g, g.vars) for g in inventory.groups])
    print(yaml.safe_dump_all([f for f in context.run()], sort_keys=False))


if __name__ == '__main__':
    _Main()
