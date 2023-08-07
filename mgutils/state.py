
from collections import Counter, defaultdict, deque
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Set, Sequence


from .logic import Action, Placeholder, Proposition, Rule, Signature, Variable, unique_product


def _check_type_conflict(name, old_type, new_type):
    if old_type != new_type:
        raise ValueError("Conflicting types for `{}`: have `{}` and `{}`.".format(name, old_type, new_type))


class State:
    """
    The current state of a world.
    """

    def __init__(self, logic: GameLogic, facts: Iterable[Proposition] = None):
        """
        Create a State.

        Parameters
        ----------
        logic :
            The logic for this state's game.
        facts : optional
            The facts that will be true in this state.
        """

        if not isinstance(logic, GameLogic):
            raise ValueError("Expected a GameLogic, found {}".format(type(logic)))
        self._logic = logic

        self._facts = defaultdict(set)
        self._vars_by_name = {}
        self._vars_by_type = defaultdict(set)
        self._var_counts = Counter()

        if facts:
            self.add_facts(facts)

    @property
    def facts(self) -> Iterable[Proposition]:
        """
        All the facts in the current state.
        """
        for fact_set in self._facts.values():
            yield from fact_set

    def facts_with_signature(self, sig: Signature) -> Set[Proposition]:
        """
        Returns all the known facts with the given signature.
        """
        return self._facts.get(sig, frozenset())

    def add_fact(self, prop: Proposition):
        """
        Add a fact to the state.
        """

        self._facts[prop.signature].add(prop)

        for var in prop.arguments:
            self._add_variable(var)

    def add_facts(self, props: Iterable[Proposition]):
        """
        Add some facts to the state.
        """

        for prop in props:
            self.add_fact(prop)

    def remove_fact(self, prop: Proposition):
        """
        Remove a fact from the state.
        """

        self._facts[prop.signature].discard(prop)

        for var in prop.arguments:
            self._remove_variable(var)

    def remove_facts(self, props: Iterable[Proposition]):
        """
        Remove some facts from the state.
        """

        for prop in props:
            self.remove_fact(prop)

    def is_fact(self, prop: Proposition) -> bool:
        """
        Returns whether a proposition is true in this state.
        """
        return prop in self._facts[prop.signature]

    def are_facts(self, props: Iterable[Proposition]) -> bool:
        """
        Returns whether the propositions are all true in this state.
        """

        for prop in props:
            if not self.is_fact(prop):
                return False

        return True

    @property
    def variables(self) -> Iterable[Variable]:
        """
        All the variables tracked by the current state.
        """
        return self._vars_by_name.values()

    def has_variable(self, var: Variable) -> bool:
        """
        Returns whether this state is aware of the given variable.
        """
        return self._vars_by_name.get(var.name) == var

    def variable_named(self, name: str) -> Variable:
        """
        Returns the variable with the given name, if known.
        """
        return self._vars_by_name[name]

    def variables_of_type(self, type: str) -> Set[Variable]:
        """
        Returns all the known variables of the given type.
        """
        return self._vars_by_type.get(type, frozenset())

    def _add_variable(self, var: Variable):
        name = var.name
        existing = self._vars_by_name.setdefault(name, var)
        _check_type_conflict(name, existing.type, var.type)

        self._vars_by_type[var.type].add(var)
        self._var_counts[name] += 1

    def _remove_variable(self, var: Variable):
        name = var.name
        self._var_counts[name] -= 1
        if self._var_counts[name] == 0:
            del self._var_counts[name]
            del self._vars_by_name[name]
            self._vars_by_type[var.type].remove(var)

    def is_applicable(self, action: Action) -> bool:
        """
        Check if an action is applicable in this state (i.e. its preconditions are met).
        """
        return self.are_facts(action.preconditions)

    def is_sequence_applicable(self, actions: Iterable[Action]) -> bool:
        """
        Check if a sequence of actions are all applicable in this state.
        """

        # The simplest implementation would copy the state and apply all the actions, but that would waste time both in
        # the copy and the variable tracking etc.

        facts = set(self.facts)
        for action in actions:
            old_len = len(facts)
            facts.difference_update(action.preconditions)
            if len(facts) != old_len - len(action.preconditions):
                return False

            facts.update(action.postconditions)

        return True

    def apply(self, action: Action) -> bool:
        """
        Apply an action to the state.

        Parameters
        ----------
        action :
            The action to apply.

        Returns
        -------
        Whether the action could be applied (i.e. whether the preconditions were met).
        """

        if self.is_applicable(action):
            self.add_facts(action.added)
            self.remove_facts(action.removed)
            return True
        else:
            return False

    def apply_on_copy(self, action: Action) -> Optional["State"]:
        """
        Apply an action to a copy of this state.

        Parameters
        ----------
        action :
            The action to apply.

        Returns
        -------
        The copied state after the action has been applied or `None` if action
        wasn't applicable.
        """
        if not self.is_applicable(action):
            return None

        state = self.copy()
        state.apply(action)
        return state

    def all_applicable_actions(self, rules: Iterable[Rule],
                               mapping: Mapping[Placeholder, Variable] = None) -> Iterable[Action]:
        """
        Get all the rule instantiations that would be valid actions in this state.

        Parameters
        ----------
        rules :
            The possible rules to instantiate.
        mapping : optional
            An initial mapping to start from, constraining the possible instantiations.

        Returns
        -------
        The actions that can be instantiated from the given rules in this state.
        """

        for rule in rules:
            yield from self.all_instantiations(rule, mapping)

    def all_instantiations(self,
                           rule: Rule,
                           mapping: Mapping[Placeholder, Variable] = None
                           ) -> Iterable[Action]:
        """
        Find all possible actions that can be instantiated from a rule in this state.

        Parameters
        ----------
        rule :
            The rule to instantiate.
        mapping : optional
            An initial mapping to start from, constraining the possible instantiations.

        Returns
        -------
        The actions that can be instantiated from the rule in this state.
        """

        for assignment in self.all_assignments(rule, mapping):
            yield rule.instantiate(assignment)

    def all_assignments(self,
                        rule: Rule,
                        mapping: Mapping[Placeholder, Optional[Variable]] = None,
                        partial: bool = False,
                        allow_partial: Callable[[Placeholder], bool] = None,
                        ) -> Iterable[Mapping[Placeholder, Optional[Variable]]]:
        """
        Find all possible placeholder assignments that would allow a rule to be instantiated in this state.

        Parameters
        ----------
        rule :
            The rule to instantiate.
        mapping : optional
            An initial mapping to start from, constraining the possible instantiations.
        partial : optional
            Whether incomplete mappings, that would require new variables or propositions, are allowed.
        allow_partial : optional
            A callback function that returns whether a partial match may involve the given placeholder.

        Returns
        -------
        The possible mappings for instantiating the rule.  Partial mappings requiring new variables will have None in
        place of existing Variables.
        """

        if mapping is None:
            mapping = {}
        else:
            # Copy the input mapping so we can mutate it
            mapping = dict(mapping)

        used_vars = set(mapping.values())

        if partial:
            new_phs = [ph for ph in rule.placeholders if ph not in mapping]
            return self._all_assignments(new_phs, mapping, used_vars, True, allow_partial)
        else:
            # Precompute the new placeholders at every depth to avoid wasted work
            seen_phs = set(mapping.keys())
            new_phs_by_depth = []
            for pred in rule.preconditions:
                new_phs = []
                for ph in pred.parameters:
                    if ph not in seen_phs:
                        new_phs.append(ph)
                        seen_phs.add(ph)
                new_phs_by_depth.append(new_phs)

            # Placeholders uniquely found in postcondition are considered as free variables.
            free_vars = [ph for ph in rule.placeholders if ph not in seen_phs]
            new_phs_by_depth.append(free_vars)

            return self._all_applicable_assignments(rule, mapping, used_vars, new_phs_by_depth, 0)

    def _all_applicable_assignments(self,
                                    rule: Rule,
                                    mapping: Dict[Placeholder, Optional[Variable]],
                                    used_vars: Set[Variable],
                                    new_phs_by_depth: List[List[Placeholder]],
                                    depth: int,
                                    ) -> Iterable[Mapping[Placeholder, Optional[Variable]]]:
        """
        Find all assignments that would be applicable in this state.  We recurse through the rule's preconditions, at
        each level determining possible variable assignments from the current facts.
        """

        new_phs = new_phs_by_depth[depth]

        if depth >= len(rule.preconditions):
            # There are no applicability constraints on the free variables, so solve them unconstrained
            yield from self._all_assignments(new_phs, mapping, used_vars, False)
            return

        pred = rule.preconditions[depth]

        types = [self._logic.types.get(t) for t in pred.signature.types]
        for subtypes in self._logic.types.multi_subtypes(types):
            signature = Signature(pred.signature.name, [t.name for t in subtypes])
            for prop in self.facts_with_signature(signature):
                for ph, var in zip(pred.parameters, prop.arguments):
                    existing = mapping.get(ph)
                    if existing is None:
                        if var in used_vars:
                            break
                        mapping[ph] = var
                        used_vars.add(var)
                    elif existing != var:
                        break
                else:
                    yield from self._all_applicable_assignments(rule, mapping, used_vars, new_phs_by_depth, depth + 1)

                # Reset the mapping to what it was before the recursive call
                for ph in new_phs:
                    var = mapping.pop(ph, None)
                    used_vars.discard(var)

    def _all_assignments(self,
                         placeholders: List[Placeholder],
                         mapping: Dict[Placeholder, Variable],
                         used_vars: Set[Variable],
                         partial: bool,
                         allow_partial: Callable[[Placeholder], bool] = None,
                         ) -> Iterable[Mapping[Placeholder, Optional[Variable]]]:
        """
        Find all possible assignments of the given placeholders, without regard to whether any predicates match.
        """

        if allow_partial is None:
            allow_partial = lambda ph: True  # noqa: E731

        candidates = []
        for ph in placeholders:
            matched_vars = set()
            for type in self._logic.types.get(ph.type).subtypes:
                matched_vars |= self.variables_of_type(type.name)
            matched_vars -= used_vars
            if partial and allow_partial(ph):
                # Allow new variables to be created
                matched_vars.add(ph)
            candidates.append(list(matched_vars))

        for assignment in unique_product(*candidates):
            for ph, var in zip(placeholders, assignment):
                if var == ph:
                    mapping[ph] = None
                elif var not in used_vars:
                    mapping[ph] = var
                    used_vars.add(var)
                else:
                    # Distinct placeholders can't be assigned the same variable
                    break
            else:
                yield mapping.copy()

            for ph in placeholders:
                used_vars.discard(mapping.get(ph))

        for ph in placeholders:
            mapping.pop(ph, None)

    def copy(self) -> "State":
        """
        Create a copy of this state.
        """

        copy = State(self._logic)

        for k, v in self._facts.items():
            copy._facts[k] = v.copy()

        copy._vars_by_name = self._vars_by_name.copy()
        for k, v in self._vars_by_type.items():
            copy._vars_by_type[k] = v.copy()
        copy._var_counts = self._var_counts.copy()

        return copy

    def serialize(self) -> Sequence:
        """
        Serialize this state.
        """
        return [f.serialize() for f in self.facts]

    @classmethod
    def deserialize(cls, data: Sequence) -> "State":
        """
        Deserialize a `State` object from `data`.
        """
        return cls([Proposition.deserialize(d) for d in data])

    def __eq__(self, other):
        if isinstance(other, State):
            return set(self.facts) == set(other.facts)
        else:
            return NotImplemented

    def __str__(self):
        lines = ["State({"]

        for sig in sorted(self._facts.keys()):
            facts = self._facts[sig]
            if len(facts) == 0:
                continue

            lines.append("    {}: {{".format(sig))
            for fact in sorted(facts):
                lines.append("        {},".format(fact))
            lines.append("    },")

        lines.append("})")

        return "\n".join(lines)
