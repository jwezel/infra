from dataclasses import dataclass, field
from operator import attrgetter
from typing import Any, Generator, Iterable, Mapping

import config

_ALL = 'ALL'
HOST_PREFIX = '='
GROUP_PREFIX = '@'


class InventoryError(Exception):
    """
    Error related to Inventory class
    """


@dataclass
class Node:
    name: str
    vars: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)
    supers: set['Node'] = field(default_factory=set, compare=False, repr=False)

    def __hash__(self):
        return hash(self.name)

    # @snoop
    def _upSearch(self) -> Generator['Node', None, None]:
        visited: set[str] = set()
        yield self
        for super_ in self.supers:
            for node in super_._upSearch():
                if node.name not in visited:
                    yield node
                    visited.add(node.name)

    def __getitem__(self, name) -> Any:
        try:
            result = next(
                node.vars.get(name) for node in sorted(self._upSearch(), key=attrgetter('height')) if name in node.vars
            )
        except StopIteration as e:
            raise InventoryError(f'{self.name} does not have access to a variable {name}') from e
        return result

    def get(self, name, default=None) -> Any:
        try:
            return self.__getitem__(name)
        except InventoryError:
            return default


@dataclass
class Host(Node):
    pass

    def __hash__(self):
        return super().__hash__()

    def groupList(self) -> set['Group']:
        return set()

    def hostList(self) -> set['Host']:
        return {self}

    @property
    def height(self) -> int:
        return 0


@dataclass
class Group(Node):
    groups: set['Group'] = field(default_factory=set, compare=False, repr=False)
    hosts: set[Host] = field(default_factory=set, compare=False)
    _height: int = -1

    def __hash__(self):
        return super().__hash__()

    def groupList(self) -> set['Group']:
        return {self}.union(*(s.groupList() for s in self.groups if s is not self))

    def hostList(self) -> set[Host]:
        return self.hosts.union(*(g.hosts for g in self.groupList()))

    @property
    def height(self) -> int:
        if self._height < 0:
            self._height = max(g.height for g in self.groups if g is not self) + 1 if self.groups else 0
        return self._height


class Inventory:

    _groups: dict[str, Group] = {_ALL: Group(_ALL)}
    _hosts: dict[str, Host] = {}

    def __init__(self, hosts: config.Config, hostgroups: config.Config):
        """
        Build group hierarchy

        Args:
            hosts (config.Config): host config
            hostgroups (config.Config): host group config
        """
        for hostname, host in ((str(k), v) for k, v in hosts.items()):
            self._hosts[hostname] = host_ = Host(hostname, vars=host.get('vars', {}))
            for groupname in host.get('groups', []) + [_ALL]:
                if groupname not in self._groups:
                    self._groups[groupname] = Group(groupname, hosts={host_}, supers={self._groups[_ALL]})
                    self._groups[_ALL].groups.add(self._groups[groupname])
                else:
                    self._groups[groupname].hosts.add(host_)
                host_.supers.add(self._groups[groupname])
        for groupname, group in ((str(k), v) for k, v in hostgroups.items()):
            group_ = self._groups[groupname]
            if groupname not in self._groups:
                self._groups[groupname] = Group(groupname)
            group_.vars = group.get('vars', {})
            for groupname in group.get('groups', []) + [_ALL]:
                if groupname not in self._groups:
                    self._groups[groupname] = Group(groupname)
                self._groups[groupname].groups.add(group_)
            group_.supers.update(self._groups[n] for n in group.get('groups', []))

    def _hostList(self, name: str) -> set[Host]:
        """
        Get host list for single name

        Resolves ! and @ prefixes.

        Args:
            name (str): Hostname or groupname

        Returns:
            set[Host]: resolved hosts
        """
        searchList: list[Mapping[str, Host | Group]]
        if name.startswith(HOST_PREFIX):
            name = name[1:]
            searchList = [self._hosts]
        elif name.startswith(GROUP_PREFIX):
            name = name[1:]
            searchList = [self._groups]
        else:
            searchList = [self._hosts, self._groups]
        return set().union(*(s[name].hostList() if name in s else set() for s in searchList))

    def hosts(self, names: Iterable[str] | None) -> set[Host]:
        """
        Get list of Host items for a list of names

        Names can be prefixed with ! to force interpreteing the name as host only, or with
        @ to force interpreteing the name as group only. With no prefixes, hosts are searched,
        and if no name is found, groups are searched as well.

        Args:
            names (Iterable[str] | None): Names of hosts and/or groups

        Returns:
            set[Host]: result

        Raises:
            exceptions
        """
        return set.union(*(self._hostList(n) for n in names or ['ALL']))

    @property
    def groups(self):
        return self._groups.values()
