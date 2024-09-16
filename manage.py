#!/bin/env python3

"""
Manage
"""

import json
import os
import platform
import re
import shlex
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from io import StringIO
from itertools import batched
from pathlib import Path
from shutil import get_terminal_size
from tempfile import NamedTemporaryFile
from typing import Any, Generator, Sequence

import config
import fabric
import invoke
import yaml
from dns.resolver import resolve
from fabric.connection import Connection
from inventory import Host, Inventory
from invoke.context import Context
from invoke.runners import Result
from omegaconf import OmegaConf
from paramiko import AutoAddPolicy
from rich import print, traceback
from rich.table import Table

COLUMNS, LINES = get_terminal_size((128, 40))

traceback.install(show_locals=True, word_wrap=True, width=COLUMNS, code_width=COLUMNS)

yaml.representer.SafeRepresenter.add_representer(
    str,
    lambda dumper, data: dumper.represent_scalar(
        'tag:yaml.org,2002:str',
        data,
        style=[None, '|'][len(data) > 80 or any(c in data for c in '\n\r\x0c\x1d\x1e\x85\u2028\u2029')],
    ),
)

DESCRIPTION = '''Run state scripts'''
NAME_HELP = '''host name or group name (* prefix forces interpretation as host name, @ prefix as group name)'''
PLAY_HELP = '''paths of plays to run. Should be after NAME arguments or separated by -- from them.'''
CONFIG_HELP = (
    '''configuration settings in the form of CONFIGNAME=VALUE. Should be after NAME arguments or separated by -- from them.'''
)
CHECK_HELP = '''enable check mode (dry run without effect)'''
DIFF_HELP = '''show difference'''
EPILOG = '''
Configuration settings available with -s / --set:

config:     path to configuration file
hosts:      path to host inventory file
hostgroups: path to host groups file
'''
REMOTE_ROOT = '/var/tmp/manage'
TERSE_SEPARATORS = ',', ':'
_MANAGE_PREFIX = '/var/lib/manage'
REMOTE_CONFIG = fabric.Config()
REMOTE_CONFIG.runners.remote.input_sleep = 0
LOCAL_CONFIG = invoke.Config()
LOCAL_CONFIG.runners.local.input_sleep = 0

GET_CONTENTS = '''[ -e "{0}" ] && cat "{0}" || true'''
PUT_CONTENTS = '''cat > {0}'''


def CamelCase(text: str) -> str:
    """
    Convert value to camel case name

    Args:
        text (str): Text

    Returns:
        str: Camel case text
    """
    return ''.join(e.title() for e in re.split(r'\W+', text))


def YamlBool(mode: bool) -> str:
    """
    Convert bool to 'yes'/'no'

    Args:
        mode (bool): Value

    Returns:
        str: 'yes' or 'no'
    """
    return ['no', 'yes'][mode]


def Fqdn(host: Host) -> str:
    hostdomain = host.get('hostdomain')
    return f"{host.name}{'.' + hostdomain if hostdomain else ''}"


def ExecResult(result: Result | None, command: str | None = None) -> Result:
    """
    Catch None runner result

    Args:
        result (Result | None): Runner result
        command (str | None, optional): Description

    Returns:
        Result: Runner result

    Raises:
        RuntimeError: If result is None
    """
    if result is None:
        raise RuntimeError('No result from execution' + f' {command}' if command else '')
    return result


def JsonEncoded(value: Any) -> Any:
    if isinstance(value, OmegaConf):
        return OmegaConf.to_container(value)
        raise TypeError(value)


class LocalMachine(Context):
    """
    LocalMachine with dummy context manager and copy method
    """

    def __init__(self, host: str | None = None, user: str | None = None, *args, **kwargs):
        """
        Summary

        Args:
            *unused: Description
        """
        super().__init__()
        self._user = user or os.environ.get('USER', os.environ.get('LOGNAME', 'root'))

    def upload(self, source: str | Path | list[str] | list[Path], remotePath: str | Path):
        """
        Copy files

        Args:
            source (str | Path): Source path
            remotePath (str): Destination path (not really remote)
        """
        command = 'rsync --recursive --perms --delete --mkpath {0} {1}'.format(
            ' '.join(str(s) for s in source) if isinstance(source, Sequence) else str(source), str(remotePath)
        )
        ExecResult(
            self.sudo(
                command,
                user=self._user,
            ),
            command,
        )

    def run(self, command, **kwargs) -> Result:
        """
        Do sudo with user given at construction

        Args:
            *args: invoke.runners.Runner.run args
            **kwargs: invoke.runners.Runner.run kwargs
        """
        return ExecResult(super().sudo(f"bash -c {shlex.quote(command)}", user=self._user, **kwargs), command)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class RemoteMachine(Connection):
    """
    A ParamikoMachine with embedded copy method
    """

    def __init__(self, host: str | None = None, user: str | None = None, *args, **kwargs):
        super().__init__(host, user, *args, **kwargs)
        self._user = user

    def upload(self, sources: str | Path | list[str] | list[Path], remotePath: str | Path):
        """
        Copy files

        Args:
            source (str | Path): Source path (local)
            remotePath (str): Destination path on remote side
        """
        with NamedTemporaryFile() as tmpfile:
            self.local(
                f'tar --create --xz --file={tmpfile.name} '
                f'{' '.join(str(s) for s in (sources if isinstance(sources, Sequence) else [sources]))}'
            )
            rtmpfile = self.run('mktemp', hide=True).stdout.rstrip('\n')
            self.put(tmpfile.name, rtmpfile)
            self.run('mkdir --parents {0};' 'tar --directory {0} --extract --xz --file={1}'.format(str(remotePath), rtmpfile))


CONNECTION: dict[Any, tuple[type[RemoteMachine], fabric.Config] | tuple[type[LocalMachine], invoke.Config]] = {
    None: (RemoteMachine, REMOTE_CONFIG),  # Serves as the default
    'remote': (RemoteMachine, REMOTE_CONFIG),
    'local': (LocalMachine, LOCAL_CONFIG),
}


class Play:
    """
    Collection of artifacts of which some can be scripts to run
    """

    _designation: str
    _artifacts: dict[Path, str]
    _scripts: list[Path]

    def __init__(self, play: str) -> None:
        """
        Initialize a Play object

        Args:
            play (str): Play
        """
        self._designation = play
        directory, _, script = play.partition(':')
        if script:
            dirPath = Path(directory)
            self._artifacts = {dirPath: self._checkSum(dirPath)}
            self._scripts = [dirPath / script]
        elif (dirPath := Path(play)).is_dir():
            self._artifacts = {dirPath: self._checkSum(dirPath)}
            self._scripts = sorted(dirPath / p for p in dirPath.iterdir() if p.is_file())
        else:
            playPath = Path(play)
            self._artifacts = {playPath: self._checkSum(playPath)}
            self._scripts = [playPath]

    @classmethod
    def _checkSum(cls, artifact: Path) -> str:
        """
        Checksum over the contents of an artifact

        For directory trees, only directory status is considered for efficiency reasons.

        Args:
            artifact (Path): Artifact path

        Returns:
            str: Check sum
        """
        sum = sha256()
        if artifact.is_dir():
            for dir_, dirs, files in artifact.walk():
                dirs.sort()
                files.sort()
                for d in dirs + files:
                    sum.update((d + str((Path(dir_) / d).stat())).encode('utf-8'))
        else:
            sum.update((str(artifact) + str(artifact.stat())).encode('utf-8'))
        return sum.hexdigest()

    @property
    def artifacts(self):
        return self._artifacts

    @property
    def scripts(self):
        return self._scripts

    @property
    def designation(self):
        return self._designation


@dataclass
class _Context:
    """
    Context
    """

    _args: Namespace
    _config: config.Config
    _hosts: set[Host]
    _hostgroups: config.Config
    _hostlist: set[Host]
    _localhost: str = resolve(platform.node(), search=True).canonical_name.to_text().rstrip('.')

    def run(self) -> Generator[dict[str, list[dict[str, Any]]], None, None]:
        """
        Run plays

        Args:
            hosts (set[Host]): hosts to target
        """
        futures: list[Future] = []
        with ThreadPoolExecutor(max_workers=self._config.get('threads')) as executor:
            # Create status for local artifacts
            plays = [Play(p) for p in self._args.play]
            # Run plays
            for hosts in batched(self._hostlist, self._config.batchsize):
                for host in hosts:
                    keyfile = host.get('keyfile')
                    if not keyfile:
                        localuser = os.environ["USER"]
                        keydir = self._config.get('keydir') or f'/home/{localuser}/.ssh'
                        user = host.get('sshuser') or localuser
                        keyfile = f"{keydir}/{user}@{Fqdn(host)}"
                    futures.append(executor.submit(self._runPlayList, host, plays, keyfile))
        return (future.result() for future in as_completed(futures))

    def _runPlayList(self, host: Host, plays: list[Play], keyfile: str) -> dict[str, list[dict[str, Any]]]:
        results: dict[str, list[dict[str, Any]]] = {}
        connection, config = CONNECTION[host.get('connection')]
        with connection(
            Fqdn(host),
            user=host.get('sshuser', os.environ.get('USER', os.environ.get('LOGNAME', 'root'))),
            port=host['sshport'],
            connect_kwargs=dict(key_filename=keyfile),
            config=config,
        ) as remote:
            for play in plays:
                remotePath = f'{REMOTE_ROOT}/{self._localhost}'
                for artifact, checkSum in play.artifacts.items():
                    remoteArtifactStatPath = f'{remotePath}/artifactstat/{artifact.name}'
                    if remote.run(GET_CONTENTS.format(remoteArtifactStatPath), hide=True).stdout != checkSum:
                        remote.run(f'mkdir --parents {remotePath}/artifactstat')
                        remote.run(PUT_CONTENTS.format(remoteArtifactStatPath), in_stream=StringIO(checkSum))
                        print(f'Uploading {artifact} to {host.name}:{str(remotePath)}')
                        remote.upload(artifact, f'{str(remotePath)}/artifacts')
                for script in play.scripts:
                    remoteScript = f'{remotePath}/artifacts/{script}'
                    print(f'Running {play.designation} on {host.name}')
                    xe = remote.run(
                        f'{remoteScript}',
                        in_stream=StringIO(
                            json.dumps(
                                dict(
                                    vars=OmegaConf.to_container(host.vars),
                                    config=OmegaConf.to_container(self._config),
                                    hosts={h.name: pp(h.asDict()) for h in self._hosts},
                                    hostgroups=OmegaConf.to_container(self._hostgroups),
                                    hostlist=sorted(h.name for h in self._hostlist),
                                    host=host.name,
                                ),
                                separators=TERSE_SEPARATORS,
                                default=JsonEncoded,
                            )
                        ),
                        env=dict(
                            PYTHONPYCACHEPREFIX=f'{remotePath}/.pycache',
                            _MANAGE_PREFIX=_MANAGE_PREFIX,
                            _MANAGE_OUTPUT_PATH=f'{remotePath}/output',
                            _MANAGE_CHECK_MODE=YamlBool(self._args.check),
                            _MANAGE_DIFF_MODE=YamlBool(self._args.difference),
                        ),
                        hide='stdout',
                    )
                    results[play.designation] = [
                        (
                            dict(status=xe.return_code, out=xe.stdout, err=xe.stderr)
                            if xe
                            else dict(status=1, out='', err='No status returned')
                        )
                    ]
        return results


def _Main() -> None:
    """
    Main
    """
    argp = ArgumentParser(
        description=DESCRIPTION, allow_abbrev=True, epilog=EPILOG, formatter_class=RawDescriptionHelpFormatter
    )
    argp.add_argument('name', nargs='*', default=['ALL'], metavar='NAME', help=NAME_HELP)
    argp.add_argument('-p', '--play', nargs='*', metavar='PLAY', help=PLAY_HELP)
    argp.add_argument('-s', '--set', dest='config', nargs='*', help=CONFIG_HELP)
    argp.add_argument('-c', '--check', action='store_true', help=CHECK_HELP)
    argp.add_argument('-d', '--difference', action='store_true', help=DIFF_HELP)
    args = argp.parse_args()
    config.Init(args.config)
    inventory = Inventory(config.Hosts, config.HostGroups)
    # config.HostGroups.ALL.vars.config = config.Settings
    # config.HostGroups.ALL.vars.hostlist = sorted(h.name for h in inventory.hosts(args.name))
    # config.HostGroups.ALL.vars.hosts = config.Hosts
    # config.HostGroups.ALL.vars.hostgroups = config.HostGroups
    context = _Context(args, config.Settings, inventory.hosts(), config.HostGroups, inventory.hosts(args.name))
    # print([(g, g.vars) for g in inventory.groups])
    # print({h: h.vars for h in inventory.hosts()})
    results = list(context.run())
    # print(results)
    # print(yaml.safe_dump_all(results, sort_keys=False))
    items: list[dict[str, Any]] = []
    for runResults in results:
        for play, playResults in runResults.items():
            for scriptResult in playResults:
                if scriptResult['status']:
                    items.append(dict(exception=True, status=scriptResult['status']))
                else:
                    tasks = json.loads(scriptResult['out'])
                    for taskNr, task in enumerate(tasks):
                        items.append(
                            dict(
                                taskNr=taskNr,
                                script=play,
                                host=task.get('host'),
                                ok=not task['ansible_result'].get('failed', False),
                                failed=task['ansible_result'].get('failed', False),
                                changed=task['ansible_result'].get('changed', False),
                                skipped=task['ansible_result'].get('skipped', False),
                                runtime=f"{task['runtime']:.3f}",
                                msg=task['ansible_result'].get('msg', ''),
                            ),
                        )
    columns = (
        ('host', 'left', None),
        ('script', 'left', None),
        ('taskNr', 'right', 'Task'),
        ('ok', 'left', 'OK'),
        ('failed', 'left', None),
        ('changed', 'left', None),
        ('skipped', 'left', None),
        ('runtime', 'right', None),
        ('msg', 'left', 'Message'),
    )
    table = Table(title='Results')
    for name, justify, title in columns:
        table.add_column(title or name.title(), justify=justify)  # type: ignore[arg-type]
    for item in items:
        table.add_row(
            *(str(item[name]) for name, _, _ in columns),
            style="cyan" if item['skipped'] else 'red' if item['failed'] else 'yellow' if item['changed'] else 'green',
        )
    print(table)


if __name__ == '__main__':
    _Main()
