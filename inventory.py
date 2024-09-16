from dataclasses import dataclass, field
import dataclasses
from operator import attrgetter
from typing import Any, Generator, Iterable, Mapping
from weakref import ReferenceType, ref

import config
from more_itertools import flatten, unique
from omegaconf import DictConfig, OmegaConf

_ALL = 'ALL'
HOST_PREFIX = '='
GROUP_PREFIX = '@'


class InventoryError(Exception):
    """
    Error related to Inventory class
    """


def Some(value: Any) -> Any:
    assert value
    return value


@dataclass(order=True)
class Node:
    name: str
    _vars: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)
    supers: set[ReferenceType['Node']] = field(default_factory=set, compare=False, repr=False)

    def __hash__(self):
        return hash(self.name)

    # @snoop
    def _upSearch(self) -> Generator['Node', None, None]:
        visited: set[str] = set()
        yield self
        for super_ in self.supers:
            for node in Some(super_())._upSearch():
                if node.name not in visited:
                    yield node
                    visited.add(node.name)

    def __getitem__(self, name) -> Any:
        try:
            result = next(
                node._vars.get(name) for node in sorted(self._upSearch(), key=attrgetter('height')) if name in node._vars
            )
        except StopIteration as e:
            raise InventoryError(f'{self.name} does not have access to a variable {name}') from e
        return result

    def get(self, name: str, default: Any = None) -> Any:
        """
        Get host variable with optional default

        Args:
            name (str): Variable name
            default (Any, optional): Default if name is not found

        Returns:
            Any: Variable value
        """
        try:
            return self.__getitem__(name)
        except InventoryError:
            return default

    @property
    def vars(self) -> DictConfig:
        """
        Get all host variables

        Returns:
            dict[str, Any]: Host variables
        """
        varList = list(flatten(list(node._vars.items()) for node in sorted(self._upSearch(), key=attrgetter('height'))))
        vars: dict[str, Any] = {}
        for _, pos in unique((v, i) for i, v in enumerate(v[0] for v in varList)):
            vars.setdefault(*varList[pos])
        return OmegaConf.create(vars)

    def asDict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        del result['supers'], result['_vars']
        result['vars'] = OmegaConf.to_container(self.vars)
        return {k: v for k, v in result.items()}


@dataclass
class Host(Node):
    pass

    def __hash__(self):
        return hash(self.name)

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
        return hash(self.name)

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
            self._hosts[hostname] = host_ = Host(hostname, _vars=host.get('vars', {}))
            for groupname in host.get('groups', []) + [_ALL]:
                if groupname not in self._groups:
                    self._groups[groupname] = Group(groupname, hosts={host_}, supers={ref(self._groups[_ALL])})
                    self._groups[_ALL].groups.add(self._groups[groupname])
                else:
                    self._groups[groupname].hosts.add(host_)
                host_.supers.add(ref(self._groups[groupname]))
        for groupname, group in ((str(k), v) for k, v in hostgroups.items()):
            group_ = self._groups[groupname]
            if groupname not in self._groups:
                self._groups[groupname] = Group(groupname)
            group_._vars = group.get('vars', {})
            for groupname in group.get('groups', []) + [_ALL]:
                if groupname not in self._groups:
                    self._groups[groupname] = Group(groupname)
                self._groups[groupname].groups.add(group_)
            group_.supers.update(ref(self._groups[n]) for n in group.get('groups', []))

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

    def hosts(self, names: Iterable[str] | None = None) -> set[Host]:
        """
        Get set of Host items for a list of names

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
        return set.union(*(self._hostList(n) for n in (names or ['ALL'])))

    @property
    def groups(self):
        return self._groups.values()
