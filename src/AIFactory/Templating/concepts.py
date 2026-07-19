import random
#
# from abc import ABCMeta
# To-use

class Concepts: # metaclass=ABCMeta
    """
    Defines the methods to call from the template engine
    """

    @property
    def __map__(self: "Concepts") -> dict:
        return getattr(self, "__map", None)

    @__map__.setter
    def __map__(self, value: dict):
        setattr(self, "__map", value)

    @__map__.deleter
    def __map__(self):
        """
        It will fail.
        :raises RuntimeError: Can't delete __map__.
        """
        raise RuntimeError("Can't delete __map__.")

    def __init__(self, function_map: list, **arguments):
        """
        Params:
            function_map (list): list of functions to call.

            arguments (kwargs): Can be given to configurate the pre-made methods.
        """
        self.hist: list|None = None
        if not isinstance(function_map, list):
            raise TypeError("function_map must be a list")
        self.arguments = arguments
        self.__map = []
        __map__ = {
            self.integer.__name__: self.integer,
            self.float_.__name__: self.float_,
            self.boolean.__name__: self.boolean,
            self.from_list.__name__: self.from_list,
            self._.__name__: self._
        }
        for func in function_map:
            __map__.setdefault(func.__name__, func)
        self.__map__ = __map__

    def extend_map(self, function_map: list):
        mapped = self.__map__
        for func in function_map:
            mapped.setdefault(func.__name__, func)

    def integer(self):
        """
        define int_start and int_end at init to customize.
        :return: an integer
        """
        start = self.arguments.get("int_start")
        end = self.arguments.get("int_end")
        return str(random.randint(start or 0, end or 1000))

    def float_(self):
        """
        define float_start and float_end at init to customize.
        :return: a float
        """
        start = self.arguments.get("float_start")
        end = self.arguments.get("float_end")
        return str(random.uniform(start or 0, end or 1000))

    def from_list(self):
        """
        define from_list at init to customize.
        :return: an item from a list
        """
        lst = self.arguments.get("from_list")
        return str(random.choice(lst))

    def boolean(self):
        """
        :return: true or false
        """
        return random.choice([True, False])

    def _(self):
        """
        :return: A ${.
        """
        return "${"

    def history(self, adding_to_history: str|None=None, index_behind_newest: int|str = 0):
        # Handling adding to history

        if adding_to_history is not None and not isinstance(adding_to_history, str):
            raise TypeError("adding_to_history must be a string")

        self.hist.append(adding_to_history)

        # Handling reading from history
        if isinstance(index_behind_newest, str):
            if index_behind_newest.isdigit():
                index_behind_newest = int(index_behind_newest)
            else:
                raise TypeError(f"Expecting an integer or a string with digits, got {index_behind_newest}")

        if index_behind_newest < 0:
            raise TypeError("index_behind_newest must be positive")

        if len(self.hist) - 1 < index_behind_newest:
            raise ValueError(f"Cannot get item {index_behind_newest} "
                             f"when the history contains only {len(self.hist)} items")

        return self.hist[len(self.hist) - 1 - index_behind_newest]