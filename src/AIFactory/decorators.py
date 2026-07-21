from typing import Callable

from AIFactory.Exceptions import InDevelopmentException


def indev(class_or_function, name) -> Callable:
    def func(self=None, *_, **__):
        _self = self # to stop IDE yelling.
        raise InDevelopmentException(f"{class_or_function} {name} is in development, please have patience.")
    return func
