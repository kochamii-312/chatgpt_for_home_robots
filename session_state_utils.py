class PrefixedSessionState:
    """Wrap an existing SessionState and prefix all keys.

    By assigning `st.session_state = PrefixedSessionState(st.session_state, "pre_")`
    each Streamlit page can maintain its own namespace without key collisions.
    """
    def __init__(self, base_state, prefix: str):
        super().__setattr__('__base', base_state)
        super().__setattr__('prefix', prefix)

    def _k(self, key: str) -> str:
        return f"{self.prefix}{key}"

    # Dictionary-style access
    def __getitem__(self, key):
        return self.__base[self._k(key)]

    def __setitem__(self, key, value):
        self.__base[self._k(key)] = value

    def __contains__(self, key):
        return self._k(key) in self.__base

    def get(self, key, default=None):
        return self.__base.get(self._k(key), default)

    def pop(self, key, default=None):
        return self.__base.pop(self._k(key), default)

    # Attribute-style access
    def __getattr__(self, key):
        if key in {'__base', 'prefix'}:
            return super().__getattribute__(key)
        return self.__base[self._k(key)]

    def __setattr__(self, key, value):
        if key in {'__base', 'prefix'}:
            super().__setattr__(key, value)
        else:
            self.__base[self._k(key)] = value
