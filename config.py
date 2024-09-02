"""
Configuration
"""

import os

from omegaconf import DictConfig, ListConfig, OmegaConf

Config = DictConfig# | ListConfig


def _Config(topic: str, config=None, prepend=None, append=None) -> Config:
    path = next(
        (
            p
            for p in (
                ([config[topic.lower()]] if config and config.get(topic.lower()) else [])
                + (prepend or [])
                + OmegaConf.create(
                    dict(
                        configpath=[
                            # First consider path from INFRA_<TOPIC> environment
                            f"${{oc.env:INFRA_{topic.upper()},''}}",
                            # Then consider the following paths:
                            f"{topic.lower()}.yaml",
                            f"{topic.lower()}",
                            f"${{oc.env:HOME}}/.config/infra/{topic.lower()}.yaml",
                            f"${{oc.env:HOME}}/.config/infra/{topic.lower()}",
                        ]
                    )
                ).configpath
                + (append or [])
            )
            if os.path.exists(p)
        ),
        None,
    )
    if path:
        return OmegaConf.load(path)
    else:
        return OmegaConf.create()

Settings = OmegaConf.create(dict(batchsize=5))
def Init(args: list[str]):
    global _CliConfig, Settings, Hosts, HostGroups
    _CliConfig = OmegaConf.from_cli(args)
    Settings.merge_with(_Config('config', config=_CliConfig))
    Settings.merge_with(_CliConfig)
    Hosts = _Config('hosts', config=Settings)
    HostGroups = _Config('hostgroups', config=Settings)

_CliConfig = Hosts = HostGroups = OmegaConf.create()
