
class InDevelopmentException(BaseException):

    def __init__(self, class_or_function, name) -> None:
        super().__init__(f"{class_or_function} {name} is in development, please have patience.")