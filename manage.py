#!/bin/env python3

"""
Manage
"""

import os
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from typing import Any

import config
import yaml
from dns import resolver
from inventory import Host, Inventory
from paramiko import WarningPolicy

# from plumbum import SshMachine
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


def runPlay(host: Host, keyfile: str, plays: list[Path]):
    with ParamikoMachine(host.name, port=host['sshport'] or 22, keyfile=keyfile, missing_host_policy=WarningPolicy) as remote:
        for play in plays:
            remote.upload(play, '/tmp')
            return remote[remote.path('/tmp') / play].run()


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
        print(f"Running {self._hosts}")
        plays: list[Path] = []
        results: list[tuple[Host, Future]] = []
        for play in self._args.play:
            playPath = Path(play)
            if playPath.is_dir():
                plays.extend(sorted(playPath.iterdir()))
            elif playPath.exists():
                plays.append(Path(play))
        with ThreadPoolExecutor(max_workers=self._config.get('threads')) as executor:
            for hosts in batched(self._hosts, self._config.batchsize):
                for host in hosts:
                    keyfile = host.vars.get('keyfile')
                    if not keyfile:
                        localuser = os.environ["USER"]
                        keydir = self._config.get('keydir') or f'/home/{localuser}/.ssh'
                        user = host.vars.get('user') or localuser
                        hostdomain = host.vars.get('hostdomain') or resolver.resolve(
                            host.name, search=True
                        ).canonical_name.to_text().rstrip('.')
                        keyfile = f"{keydir}/{user}@{hostdomain}"
                    results.append((host, executor.submit(runPlay, host, keyfile, plays)))
        return [
            {h.name: dict(zip(['status', 'stdout', 'stderr'], r.result()))}
            for h, r in zip((r[0] for r in results), as_completed(r[1] for r in results))
        ]


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
    print(yaml.safe_dump_all([f for f in context.run()]))


if __name__ == '__main__':
    _Main()
