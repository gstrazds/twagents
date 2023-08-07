# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


from collections import Counter, defaultdict, deque
from functools import total_ordering, lru_cache
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Set, Sequence

try:
    from typing import Collection
except ImportError:
    # Collection is new in Python 3.6 -- fall back on Iterable for 3.5
    from typing import Iterable as Collection


from mementos import memento_factory, with_metaclass


def uniquify(seq):
    """ Order preserving uniquify.

    References
    ----------
    Made by Dave Kirby
    https://www.peterbe.com/plog/uniqifiers-benchmark
    """
    seen = set()
    return [x for x in seq if x not in seen and not seen.add(x)]


def unique_product(*iterables):
    """ Cartesian product of input iterables with pruning.

    This method prunes any product tuple with duplicate elements in it.

    Example:
        unique_product('ABC', 'Ax', 'xy') --> Axy BAx BAy Bxy CAx CAy Cxy

    Notes:
        This method is faster than the following equivalent code:

        >>> for result in itertools.product(*args):
        >>>     if len(set(result)) == len(result):
        >>>         yield result

    """
    _SENTINEL = object()

    def _unique_product_recursive(pools, result, i):
        if i >= len(pools):
            yield tuple(result)
            return

        for e in pools[i]:
            if e not in result:
                result[i] = e
                yield from _unique_product_recursive(pools, result, i + 1)
                result[i] = _SENTINEL

    pools = [tuple(pool) for pool in iterables]
    result = [_SENTINEL] * len(pools)
    return _unique_product_recursive(pools, result, 0)


# We use first-order logic to represent the state of the world, and the actions
# that can be applied to it.  The relevant classes are:
#
# - Variable: a logical variable representing an entity in the world
#
# - Proposition: a predicate applied to some variables, e.g. in(cup, kitchen)
#
# - Action: an action that modifies the state of the world, with propositions as
#   pre-/post-conditions
#
# - State: holds the set of factual propositions in the current world state
#
# - Placeholder: a formal parameter to a predicate
#
# - Predicate: an unevaluated predicate, e.g. in(object, container)
#
# - Rule: a template for an action, with predicates as pre-/post-conditions

# Performance note: many of these classes are performance-critical.  The
# optimization techniques used in their implementation include:
#
# - Immutability, which enables heavy object sharing
#
# - Using __slots__ to save memory and speed up attribute access
#
# - For classes that appear as dictionary keys or in sets, we cache the hash
#   code in the _hash field
#
# - For those same classes, we implement __eq__() like this:
#       return self.attr1 == other.attr1 and self.attr2 == other.attr2
#   rather than like this:
#       return (self.attr1, self.attr2) == (other.attr1, other.attr2)
#   to avoid allocating tuples
#
# - List comprehensions are preferred to generator expressions


@total_ordering
class Type:
    """
    A variable type.
    """

    def __init__(self, name: str, parents: Iterable[str]):
        self.name = name
        self.parents = tuple(parents)

    def _attach(self, hier: "TypeHierarchy"):
        self._hier = hier

    @property
    def parent_types(self) -> Iterable["Type"]:
        """
        The parents of this type as Type objects.
        """
        return (self._hier.get(name) for name in self.parents)

    @property
    def ancestors(self) -> Iterable["Type"]:
        """
        The ancestors of this type (not including itself).
        """
        return self._hier.closure(self, lambda t: t.parent_types)

    @property
    def supertypes(self) -> Iterable["Type"]:
        """
        This type and its ancestors.
        """
        yield self
        yield from self.ancestors

    def is_supertype_of(self, other: "Type") -> bool:
        return self in other.supertypes

    def has_supertype_named(self, name: str) -> bool:
        return self._hier.get(name).is_supertype_of(self)

    @property
    def children(self) -> Iterable[str]:
        """
        The names of the direct children of this type.
        """
        return self._hier._children[self.name]

    @property
    def child_types(self) -> Iterable["Type"]:
        """
        The direct children of this type.
        """
        return (self._hier.get(name) for name in self.children)

    @property
    def descendants(self) -> Iterable["Type"]:
        """
        The descendants of this type (not including itself).
        """
        return self._hier.closure(self, lambda t: t.child_types)

    @property
    def subtypes(self) -> Iterable["Type"]:
        """
        This type and its descendants.
        """
        yield self
        yield from self.descendants

    def is_subtype_of(self, other: "Type") -> bool:
        return self in other.subtypes

    def has_subtype_named(self, name: str) -> bool:
        return self._hier.get(name).is_subtype_of(self)

    def __str__(self):
        if self.parents:
            return "{} : {}".format(self.name, ", ".join(self.parents))
        else:
            return self.name

    def __repr__(self):
        return "Type({!r}, {!r})".format(self.name, self.parents)

    def __eq__(self, other):
        if isinstance(other, Type):
            return self.name == other.name
        else:
            return NotImplemented

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        if isinstance(other, Type):
            return self.name < other.name
        else:
            return NotImplemented


class TypeHierarchy:
    """
    A hierarchy of types.
    """

    def __init__(self):
        self._types = {}
        self._children = defaultdict(list)
        self._cache = {}

    def add(self, type: Type):
        if type.name in self._types:
            raise ValueError("Duplicate type {}".format(type.name))

        type._attach(self)
        self._types[type.name] = type

        for parent in type.parents:
            children = self._children[parent]
            children.append(type.name)
            children.sort()

        # Adding a new type invalidates the cache.
        self._cache = {}

    def get(self, name: str) -> Type:
        return self._types[name]

    def __iter__(self):
        yield from self._types.values()

    def __len__(self):
        return len(self._types)

    def closure(self, type: Type, expand: Callable[[Type], Iterable[Type]]) -> Iterable[Type]:
        r"""
        Compute the transitive closure in a type lattice according to some type
        relationship (generally direct sub-/super-types).

        Such a lattice may look something like this::

              A
             / \
            B   C
             \ /
              D

        so the closure of D would be something like [B, C, A].
        """

        return self._bfs_unique(type, expand)

    def _multi_expand(self, types: Collection[Type], expand: Callable[[Type], Iterable[Type]]) -> Iterable[Collection[Type]]:
        """
        Apply the expand() function to every element of a type sequence in turn.
        """

        for i in range(len(types)):
            expansion = list(types)
            for replacement in expand(expansion[i]):
                expansion[i] = replacement
                yield tuple(expansion)

    def multi_closure(self, types: Collection[Type], expand: Callable[[Type], Iterable[Type]]) -> Iterable[Collection[Type]]:
        r"""
        Compute the transitive closure of a sequence of types in a type lattice
        induced by some per-type relationship (generally direct sub-/super-types).

        For a single type, such a lattice may look something like this::

              A
             / \
            B   C
             \ /
              D

        so the closure of D would be something like [B, C, A].  For multiple
        types at once, the lattice is more complicated::

                        __ (A,A) __
                       /   |   |   \
                  (A,B) (A,C) (B,A) (C,A)
              *******************************
            (A,D) (B,B) (B,C) (C,B) (C,C) (D,A)
              *******************************
                  (B,D) (C,D) (D,B) (D,C)
                       \   |   |   /
                        \_ (D,D) _/
        """

        return self._bfs_unique(types, lambda ts: self._multi_expand(ts, expand))

    def _bfs_unique(self, start, expand):
        """
        Apply breadth-first search, returning only previously unseen nodes.
        """

        seen = set()
        queue = deque(expand(start))
        while queue:
            item = queue.popleft()
            yield item
            for expansion in expand(item):
                if expansion not in seen:
                    seen.add(expansion)
                    queue.append(expansion)

    def multi_ancestors(self, types: Collection[Type]) -> Iterable[Collection[Type]]:
        """
        Compute the ancestral closure of a sequence of types.  If these are the
        types of some variables, the result will be all the function parameter
        types that could also accept those variables.
        """
        return self.multi_closure(types, lambda t: t.parent_types)

    def multi_supertypes(self, types: Collection[Type]) -> Iterable[Collection[Type]]:
        """
        Computes the ancestral closure of a sequence of types, including the
        initial types.
        """
        yield tuple(types)
        yield from self.multi_ancestors(types)

    def multi_descendants(self, types: Collection[Type]) -> Iterable[Collection[Type]]:
        """
        Compute the descendant closure of a sequence of types.  If these are the
        types of some function parameters, the result will be all the variable
        types that could also be passed to this function.
        """
        return self.multi_closure(types, lambda t: t.child_types)

    def multi_subtypes(self, types: Collection[Type]) -> List[Collection[Type]]:
        """
        Computes the descendant closure of a sequence of types, including the
        initial types.
        """
        types = tuple(types)
        if types not in self._cache:
            self._cache[types] = [types] + list(self.multi_descendants(types))

        return self._cache[types]


@total_ordering
class Variable:
    """
    A variable representing an object in a world.
    """

    __slots__ = ("name", "type", "_hash")

    def __init__(self, name: str, type: Optional[str] = None):
        """
        Create a Variable.

        Parameters
        ----------
        name :
            The (unique) name of the variable.
        type : optional
            The type of the variable.  Defaults to the same as the name.
        """

        self.name = name

        if type is None:
            type = name
        self.type = type

        self._hash = hash((self.name, self.type))

    def is_a(self, type: Type) -> bool:
        return type.has_subtype_named(self.type)

    def __str__(self):
        if self.type == self.name:
            return self.name
        else:
            return "{}: {}".format(self.name, self.type)

    def __repr__(self):
        return "Variable({!r}, {!r})".format(self.name, self.type)

    def __eq__(self, other):
        if isinstance(other, Variable):
            return self.name == other.name and self.type == other.type
        else:
            return NotImplemented

    def __hash__(self):
        return self._hash

    def __lt__(self, other):
        if isinstance(other, Variable):
            return (self.name, self.type) < (other.name, other.type)
        else:
            return NotImplemented

    def serialize(self) -> Mapping:
        return {
            "name": self.name,
            "type": self.type,
        }

    @classmethod
    def deserialize(cls, data: Mapping) -> "Variable":
        return cls(data["name"], data["type"])


SignatureTracker = memento_factory(
    'SignatureTracker',
    lambda cls, args, kwargs: (
        cls,
        kwargs.get("name", args[0] if len(args) >= 1 else None),
        tuple(kwargs.get("types", args[1] if len(args) == 2 else []))
    )
)


@total_ordering
class Signature(with_metaclass(SignatureTracker, object)):
    """
    The type signature of a Predicate or Proposition.
    """

    __slots__ = ("name", "types", "_hash")

    def __init__(self, name: str, types: Iterable[str]):
        """
        Create a Signature.

        Parameters
        ----------
        name :
            The name of the proposition/predicate this signature is for.
        types :
            The types of the parameters to the proposition/predicate.
        """

        self.name = name
        self.types = tuple(types)
        self._hash = hash((self.name, self.types))

    def __str__(self):
        return "{}({})".format(self.name, ", ".join(map(str, self.types)))

    def __repr__(self):
        return "Signature({!r}, {!r})".format(self.name, self.types)

    def __eq__(self, other):
        if isinstance(other, Signature):
            return self.name == other.name and self.types == other.types
        else:
            return NotImplemented

    def __hash__(self):
        return self._hash

    def __lt__(self, other):
        if isinstance(other, Signature):
            return (self.name, self.types) < (other.name, other.types)
        else:
            return NotImplemented


PropositionTracker = memento_factory(
    'PropositionTracker',
    lambda cls, args, kwargs: (
        cls,
        kwargs.get("name", args[0] if len(args) >= 1 else None),
        tuple(v.name for v in kwargs.get("arguments", args[1] if len(args) == 2 else []))
    )
)


@total_ordering
class Proposition(with_metaclass(PropositionTracker, object)):
    """
    An instantiated Predicate, with concrete variables for each placeholder.
    """

    __slots__ = ("name", "arguments", "signature", "_hash")

    def __init__(self, name: str, arguments: Iterable[Variable] = []):
        """
        Create a Proposition.

        Parameters
        ----------
        name :
            The name of the proposition.
        arguments :
            The variables this proposition is applied to.
        """

        self.name = name
        self.arguments = tuple(arguments)
        self.signature = Signature(name, [var.type for var in self.arguments])
        self._hash = hash((self.name, self.arguments))

    @property
    def names(self) -> Collection[str]:
        """
        The names of the variables in this proposition.
        """
        return tuple([var.name for var in self.arguments])

    @property
    def types(self) -> Collection[str]:
        """
        The types of the variables in this proposition.
        """
        return self.signature.types

    def __str__(self):
        return "{}({})".format(self.name, ", ".join(map(str, self.arguments)))

    def __repr__(self):
        return "Proposition({!r}, {!r})".format(self.name, self.arguments)

    def __eq__(self, other):
        if isinstance(other, Proposition):
            return self.name == other.name and self.arguments == other.arguments
        else:
            return NotImplemented

    def __hash__(self):
        return self._hash

    def __lt__(self, other):
        if isinstance(other, Proposition):
            return (self.name, self.arguments) < (other.name, other.arguments)
        else:
            return NotImplemented

    def serialize(self) -> Mapping:
        return {
            "name": self.name,
            "arguments": [var.serialize() for var in self.arguments],
        }

    @classmethod
    def deserialize(cls, data: Mapping) -> "Proposition":
        name = data["name"]
        args = [Variable.deserialize(arg) for arg in data["arguments"]]
        return cls(name, args)


@total_ordering
class Placeholder:
    """
    A symbolic placeholder for a variable in a Predicate.
    """

    __slots__ = ("name", "type", "_hash")

    def __init__(self, name: str, type: Optional[str] = None):
        """
        Create a Placeholder.

        Parameters
        ----------
        name :
            The name of this placeholder.
        type : optional
            The type of variable represented.  Defaults to the name with any trailing apostrophes stripped.
        """

        self.name = name

        if type is None:
            type = name.rstrip("'")
        self.type = type

        self._hash = hash((self.name, self.type))

    def __str__(self):
        if self.type == self.name.rstrip("'"):
            return self.name
        else:
            return "{}: {}".format(self.name, self.type)

    def __repr__(self):
        return "Placeholder({!r}, {!r})".format(self.name, self.type)

    def __eq__(self, other):
        if isinstance(other, Placeholder):
            return self.name == other.name and self.type == other.type
        else:
            return NotImplemented

    def __hash__(self):
        return self._hash

    def __lt__(self, other):
        if isinstance(other, Placeholder):
            return (self.name, self.type) < (other.name, other.type)
        else:
            return NotImplemented

    def serialize(self) -> Mapping:
        return {
            "name": self.name,
            "type": self.type,
        }

    @classmethod
    def deserialize(cls, data: Mapping) -> "Placeholder":
        return cls(data["name"], data["type"])


@total_ordering
class Predicate:
    """
    A boolean-valued function over variables.
    """

    def __init__(self, name: str, parameters: Iterable[Placeholder]):
        """
        Create a Predicate.

        Parameters
        ----------
        name :
            The name of this predicate.
        parameters :
            The symbolic arguments to this predicate.
        """

        self.name = name
        self.parameters = tuple(parameters)
        self.signature = Signature(name, [ph.type for ph in self.parameters])

    @property
    def names(self) -> Collection[str]:
        """
        The names of the placeholders in this predicate.
        """
        return tuple([ph.name for ph in self.parameters])

    @property
    def types(self) -> Collection[str]:
        """
        The types of the placeholders in this predicate.
        """
        return self.signature.types

    def __str__(self):
        return "{}({})".format(self.name, ", ".join(map(str, self.parameters)))

    def __repr__(self):
        return "Predicate({!r}, {!r})".format(self.name, self.parameters)

    def __eq__(self, other):
        if isinstance(other, Predicate):
            return (self.name, self.parameters) == (other.name, other.parameters)
        else:
            return NotImplemented

    def __hash__(self):
        return hash((self.name, self.parameters))

    def __lt__(self, other):
        if isinstance(other, Predicate):
            return (self.name, self.parameters) < (other.name, other.parameters)
        else:
            return NotImplemented

    def serialize(self) -> Mapping:
        return {
            "name": self.name,
            "parameters": [ph.serialize() for ph in self.parameters],
        }

    @classmethod
    def deserialize(cls, data: Mapping) -> "Predicate":
        name = data["name"]
        params = [Placeholder.deserialize(ph) for ph in data["parameters"]]
        return cls(name, params)

    def substitute(self, mapping: Mapping[Placeholder, Placeholder]) -> "Predicate":
        """
        Copy this predicate, substituting certain placeholders for others.

        Parameters
        ----------
        mapping :
            A mapping from old to new placeholders.
        """

        params = [mapping.get(param, param) for param in self.parameters]
        return Predicate(self.name, params)

    def instantiate(self, mapping: Mapping[Placeholder, Variable]) -> Proposition:
        """
        Instantiate this predicate with the given mapping.

        Parameters
        ----------
        mapping :
            A mapping from Placeholders to Variables.

        Returns
        -------
        The instantiated Proposition with each Placeholder mapped to the corresponding Variable.
        """

        args = [mapping[param] for param in self.parameters]
        return Proposition(self.name, args)

    def match(self, proposition: Proposition) -> Optional[Mapping[Placeholder, Variable]]:
        """
        Match this predicate against a concrete proposition.

        Parameters
        ----------
        proposition :
            The proposition to match against.

        Returns
        -------
        The mapping from placeholders to variables such that `self.instantiate(mapping) == proposition`, or `None` if no
        such mapping exists.
        """

        if self.name != proposition.name:
            return None
        else:
            return {ph: var for ph, var in zip(self.parameters, proposition.arguments)}


class Alias:
    """
    A shorthand predicate alias.
    """

    def __init__(self, pattern: Predicate, replacement: Iterable[Predicate]):
        self.pattern = pattern
        self.replacement = tuple(replacement)

    def __str__(self):
        return "{} = {}".format(self.pattern, " & ".join(map(str, self.replacement)))

    def __repr__(self):
        return "Alias({!r}, {!r})".format(self.pattern, self.replacement)

    def expand(self, predicate: Predicate) -> Collection[Predicate]:
        """
        Expand a use of this alias into its replacement.
        """
        if predicate.signature == self.pattern.signature:
            mapping = dict(zip(self.pattern.parameters, predicate.parameters))
            return tuple([pred.substitute(mapping) for pred in self.replacement])
        else:
            return predicate


class Action:
    """
    An action in the environment.
    """

    def __init__(self, name: str, preconditions: Iterable[Proposition], postconditions: Iterable[Proposition]):
        """
        Create an Action.

        Parameters
        ----------
        name :
            The name of this action.
        preconditions :
            The preconditions that must hold before this action is applied.
        postconditions :
            The conditions that replace the preconditions once applied.
        """

        self.name = name
        self.command_template = None
        self.reverse_name = None
        self.reverse_command_template = None
        self.preconditions = tuple(preconditions)
        self.postconditions = tuple(postconditions)

        self._pre_set = frozenset(self.preconditions)
        self._post_set = frozenset(self.postconditions)

    @property
    def variables(self):
        if not hasattr(self, "_variables"):
            self._variables = tuple(uniquify(var for prop in self.all_propositions for var in prop.arguments))

        return self._variables

    @property
    def all_propositions(self) -> Collection[Proposition]:
        """
        All the pre- and post-conditions.
        """
        return self.preconditions + self.postconditions

    @property
    def added(self) -> Collection[Proposition]:
        """
        All the new propositions being introduced by this action.
        """
        return self._post_set - self._pre_set

    @property
    def removed(self) -> Collection[Proposition]:
        """
        All the old propositions being removed by this action.
        """
        return self._pre_set - self._post_set

    def __str__(self):
        # Infer carry-over preconditions for pretty-printing
        pre = []
        for prop in self.preconditions:
            if prop in self._post_set:
                pre.append("$" + str(prop))
            else:
                pre.append(str(prop))

        post = [str(prop) for prop in self.postconditions if prop not in self._pre_set]

        return "{} :: {} -> {}".format(self.name, " & ".join(pre), " & ".join(post))

    def __repr__(self):
        return "Action({!r}, {!r}, {!r})".format(self.name, self.preconditions, self.postconditions)

    def __eq__(self, other):
        if isinstance(other, Action):
            return (self.name, self._pre_set, self._post_set) == (other.name, other._pre_set, other._post_set)
        else:
            return NotImplemented

    def __hash__(self):
        return hash((self.name, self._pre_set, self._post_set))

    def serialize(self) -> Mapping:
        return {
            "name": self.name,
            "preconditions": [prop.serialize() for prop in self.preconditions],
            "postconditions": [prop.serialize() for prop in self.postconditions],
            "command_template": self.command_template,
            "reverse_name": self.reverse_name,
            "reverse_command_template": self.reverse_command_template,
        }

    @classmethod
    def deserialize(cls, data: Mapping) -> "Action":
        name = data["name"]
        pre = [Proposition.deserialize(prop) for prop in data["preconditions"]]
        post = [Proposition.deserialize(prop) for prop in data["postconditions"]]
        action = cls(name, pre, post)
        action.command_template = data.get("command_template")
        action.reverse_name = data.get("reverse_name")
        action.reverse_command_template = data.get("reverse_command_template")
        return action

    def inverse(self, name: Optional[str] = None) -> "Action":
        """
        Invert the direction of this action.

        Parameters
        ----------
        name : optional
            The new name for the inverse action.

        Returns
        -------
        An action that does the exact opposite of this one.
        """
        name = name or self.reverse_name or "r_" + self.name
        action = Action(name, self.postconditions, self.preconditions)
        action.command_template = self.reverse_command_template
        action.reverse_command_template = self.command_template
        return action

    def format_command(self, mapping: Dict[str, str] = {}):
        mapping = mapping or {v.name: v.name for v in self.variables}
        return self.command_template.format(**mapping)


class Rule:
    """
    A template for an action.
    """

    def __init__(self, name: str, preconditions: Iterable[Predicate], postconditions: Iterable[Predicate]):
        """
        Create a Rule.

        Parameters
        ----------
        name :
            The name of this rule.
        preconditions :
            The preconditions that must hold before this rule is applied.
        postconditions :
            The conditions that replace the preconditions once applied.
        """

        self.name = name
        self.command_template = None
        self.reverse_rule = None
        self._cache = {}
        self.preconditions = tuple(preconditions)
        self.postconditions = tuple(postconditions)

        self._pre_set = frozenset(self.preconditions)
        self._post_set = frozenset(self.postconditions)

        self.placeholders = tuple(uniquify(ph for pred in self.all_predicates for ph in pred.parameters))

    @property
    def all_predicates(self) -> Iterable[Predicate]:
        """
        All the pre- and post-conditions.
        """
        return self.preconditions + self.postconditions

    def __str__(self):
        # Infer carry-over preconditions for pretty-printing
        pre = []
        for pred in self.preconditions:
            if pred in self._post_set:
                pre.append("$" + str(pred))
            else:
                pre.append(str(pred))

        post = [str(pred) for pred in self.postconditions if pred not in self._pre_set]

        return "{} :: {} -> {}".format(self.name, " & ".join(pre), " & ".join(post))

    def __repr__(self):
        return "Rule({!r}, {!r}, {!r})".format(self.name, self.preconditions, self.postconditions)

    def __eq__(self, other):
        if isinstance(other, Rule):
            return (self.name, self._pre_set, self._post_set) == (other.name, other._pre_set, other._post_set)
        else:
            return NotImplemented

    def __hash__(self):
        return hash((self.name, self._pre_set, self._post_set))

    def serialize(self) -> Mapping:
        return {
            "name": self.name,
            "preconditions": [pred.serialize() for pred in self.preconditions],
            "postconditions": [pred.serialize() for pred in self.postconditions],
        }

    @classmethod
    def deserialize(cls, data: Mapping) -> "Rule":
        name = data["name"]
        pre = [Predicate.deserialize(pred) for pred in data["preconditions"]]
        post = [Predicate.deserialize(pred) for pred in data["postconditions"]]
        return cls(name, pre, post)

    def _make_command_template(self, mapping: Mapping[Placeholder, Variable]) -> str:
        if self.command_template is None:
            return None

        substitutions = {ph.name: "{{{}}}".format(var.name) for ph, var in mapping.items()}
        return self.command_template.format(**substitutions)

    def substitute(self, mapping: Mapping[Placeholder, Placeholder], name=None) -> "Rule":
        """
        Copy this rule, substituting certain placeholders for others.

        Parameters
        ----------
        mapping :
            A mapping from old to new placeholders.
        """

        if name is None:
            name = self.name
        pre_subst = [pred.substitute(mapping) for pred in self.preconditions]
        post_subst = [pred.substitute(mapping) for pred in self.postconditions]
        return Rule(name, pre_subst, post_subst)

    def instantiate(self, mapping: Mapping[Placeholder, Variable]) -> Action:
        """
        Instantiate this rule with the given mapping.

        Parameters
        ----------
        mapping :
            A mapping from Placeholders to Variables.

        Returns
        -------
        The instantiated Action with each Placeholder mapped to the corresponding Variable.
        """

        key = tuple(mapping[ph] for ph in self.placeholders)
        if key in self._cache:
            return self._cache[key]

        pre_inst = [pred.instantiate(mapping) for pred in self.preconditions]
        post_inst = [pred.instantiate(mapping) for pred in self.postconditions]
        action = Action(self.name, pre_inst, post_inst)

        action.command_template = self._make_command_template(mapping)
        if self.reverse_rule:
            action.reverse_name = self.reverse_rule.name
            action.reverse_command_template = self.reverse_rule._make_command_template(mapping)

        self._cache[key] = action
        return action

    def match(self, action: Action) -> Optional[Mapping[Placeholder, Variable]]:
        """
        Match this rule against a concrete action.

        Parameters
        ----------
        action :
            The action to match against.

        Returns
        -------
        The mapping from placeholders to variables such that `self.instantiate(mapping) == action`, or `None` if no such
        mapping exists.
        """

        if self.name != action.name:
            return None

        candidates = [action.variables] * len(self.placeholders)

        # A same variable can't be assigned to different placeholders.
        # Using `unique_product` avoids generating those in the first place.
        for assignment in unique_product(*candidates):
            mapping = {ph: var for ph, var in zip(self.placeholders, assignment)}
            if self.instantiate(mapping) == action:
                return mapping

        return None

    def inverse(self, name=None) -> "Rule":
        """
        Invert the direction of this rule.

        Parameters
        ----------
        name : optional
            The new name for the inverse rule.

        Returns
        -------
        A rule that does the exact opposite of this one.
        """

        if name is None:
            name = self.name
            if self.reverse_rule:
                name = self.reverse_rule.name

        if self.reverse_rule:
            return self.reverse_rule

        rule = Rule(name, self.postconditions, self.preconditions)
        rule.reverse_rule = self
        return rule


