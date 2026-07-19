import warnings
from typing import List, Any, Callable

class FixingHelper:
    def __init__(self,
                 is_wrongs: List[Callable[[Any], bool]],
                 descriptions: List[str],):
        if not isinstance(is_wrongs, list):
            raise TypeError("is_wrongs must be a list")
        if not isinstance(descriptions, list):
            raise TypeError("descriptions must be a list")
        if not all(callable(i) for i in is_wrongs):
            raise ValueError("is_wrongs must only contain Lambda(or any other callable) elements")
        if not all(isinstance(i, str) for i in descriptions):
            raise ValueError("descriptions must only contain strings")
        if not len(is_wrongs) == len(descriptions):
            raise ValueError("is_wrongs and descriptions must have the same length")

        self.last_thing = None
        self.is_wrongs = is_wrongs
        self.descriptions = descriptions

    def this_thing_is_garbage(self):
        headups = []

        for i in range(len(self.is_wrongs)):
            if hasattr(self.last_thing, "__iter__"):
                for item in self.last_thing:
                    if self.is_wrongs[i](item):
                        headups.append(f"Problem {len(headups) + 1}, {self.descriptions[i]}")
            else:
                if self.is_wrongs[i](self.last_thing):
                    headups.append(f"Problem {len(headups) + 1}, {self.descriptions[i]}")

        if headups:
            raising = ("Well, look at the garbage you made.\n"
                       "Don't worry, we'll try to fix that together.\n"
                       "Everyone make mistakes.\n"
                       "".join(f"\t{i}\n" for i in headups))

            raise RuntimeError(raising)
        else:
            self.last_thing = None
            warnings.warn("Well, I can't help more with that one.")

        return "Have a good day!"

    def i_updated_it_to_this_and_still_garbage(self, new_useless_garbage: str):
        if not isinstance(new_useless_garbage, str):
            raise TypeError("new_useless_garbage must be a string")

        self.last_thing = new_useless_garbage
        self.this_thing_is_garbage()