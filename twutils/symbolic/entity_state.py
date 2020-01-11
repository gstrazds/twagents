

class EntityState:
    """
    Keeps track of the current state of a thing.

    """
    _state_properties = {  # map: state property name => meta property name
        'is_open': 'openable',
        'is_locked': 'lockable',
        'is_on': 'switchable',
        'is_cooked': 'cookable',
        'is_cut': 'cuttable'
    }

    # map: meta property name => state property name
    _meta_properties = dict(map(reversed, _state_properties.items()))

    def __init__(self):
        self._state_values = {}
        self._init_states = {}

    # def remove(self):
    #     self.exists = False   # this Entity no longer exists within the world

    def reset(self, entity):
        """ reset state to what it was when we first encountered this Entity"""
        for prop_name in self._init_states:
            print(f"Reseting entity_state {entity}.[{prop_name}] = {self._init_states[prop_name]}")
            #self._state_values[prop_name] = self._init_states[prop_name]
            self._state_values[prop_name] = None  #self._init_states[prop_name]

    def __getattr__(self, prop_name):
        if self._has_prop(prop_name):
            return self._get_state_val(prop_name)
        elif EntityState._meta_properties.get(prop_name, None):
            return self._has_prop(EntityState._meta_properties[prop_name])
        print(self._state_values)
        raise AttributeError(f'{self.__class__.__name__}.{prop_name} is invalid.')

    def __setattr__(self, state_prop_name, value):
        if state_prop_name in EntityState._state_properties:
            self._set_state_val(state_prop_name, value)
        else:
            super().__setattr__(state_prop_name, value)

    def __dir__(self):
        return super().__dir__() + self._get_state_prop_names() + self._get_meta_attr_names()

    def _get_state_prop_names(self):
        return [k for k in self._state_values.keys()]

    def _get_meta_attr_names(self):
        return [self._get_meta_prop(prop) for prop in EntityState._state_properties.keys() if \
                self._get_meta_prop(prop)]

    def _get_meta_prop(self, prop_name: str):
        """e.g. _get_meta_prop('is_open') => 'openable')  or _get_meta_prop('is_locked') => 'lockable'"""
        meta_prop = EntityState._state_properties.get(prop_name, None)
        if meta_prop and self._has_prop(prop_name):
            return meta_prop
        return None

    def _has_prop(self, state_prop):
        """ state_prop should be a value from state_properties
            e.g. _has_prop('is_open') or _has_prop('is_locked')"""
        return state_prop in self._state_values

    def add_state_variable(self, metaprop, stateprop, initial_value=None):
        if metaprop in EntityState._meta_properties:
            assert stateprop == EntityState._meta_properties[metaprop]
            if self._has_prop(stateprop):
                print(f"WARNING: {self} already {metaprop}: {stateprop}={self._get_state_val(stateprop)}")
                if initial_value != self._get_state_val(stateprop):
                    if initial_value:   # Don't set it if it is indeterminate
                        print(
                            f"WARNING: overriding:{self._get_state_val(stateprop)}"
                            " in add_state_variable({metaprop}) {stateprop}={initial_value}")
                        self._set_state_val(stateprop, initial_value)
        else:
            EntityState._meta_properties[metaprop] = stateprop
            EntityState._state_properties[stateprop] = metaprop
            self._set_state_val(stateprop, initial_value)

    def _set_state_val(self, state_prop, val):
        # if not attr_name in EntityState.state_properties:
        #     print(f"ASSERTION FAILURE: unknown state attribute {attr_name}")
        #     return None
        # state_prop = EntityState.state_properties[attr_name]
        if not state_prop in EntityState._state_properties:
            print(f"WARNING: setting unknown state variable {state_prop}={val}")
            raise AttributeError(f'{self.__class__.__name__}.{state_prop} is invalid.')
        if self._init_states.get(state_prop, None) is None:
            self._init_states[state_prop] = val
        prev_val = self._state_values.get(state_prop, None)
        self._state_values[state_prop] = val
        return prev_val

    def _get_state_val(self, state_prop):
        # if not attr_name in EntityState.state_properties:
        #     print(f"ASSERTION FAILURE: unknown state attribute {attr_name}")
        #     return None
        # state_prop = EntityState.state_properties[attr_name]
        return self._state_values.get(state_prop, None)

    def open(self):  # returns previous value of 'is_open'
        return self._set_state_val('is_open', True)

    def close(self):  # returns previous value of 'is_open'
        return self._set_state_val('is_open', False)

    # @property
    # def is_open(self):
    #     return self._get_state_val('is_open')
    #
    # @is_open.setter
    # def set_is_open(self, val):
    #     self._set_state_val('is_open', val)

    def lock(self):
        return self._set_state_val('is_locked', True)

    def unlock(self):
        return self._set_state_val('is_locked', False)

    # def switchable(self):

    def turn_on(self):
        return self._set_state_val('is_on', True)

    def turn_off(self):
        return self._set_state_val('is_on', False)

    # @property
    # def cookable(self):

    def cook(self, cooked_state='cooked'):
        return self._set_state_val('is_cooked', cooked_state)

    def not_cooked(self):
        return self._set_state_val('is_cooked', '')  # equiv to False, but can also be tested with str.startswith()

    # @property
    # def cuttable(self):

    def cut(self, cut_state='cut'):  # 'sliced', 'chopped', 'diced', etc...
        return self._set_state_val('is_cut', cut_state)

    def not_cut(self):
        return self._set_state_val('is_cut', '')  # equiv to False, but can also be tested with str.startswith()

    def __str__(self):
        return str(self._state_values)

