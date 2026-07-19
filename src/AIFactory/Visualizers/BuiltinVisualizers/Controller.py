import datetime
from typing import Dict, Literal, List, Tuple, Iterator, Callable

from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
from torch.nn import Parameter

from transformers import TrainingArguments, TrainerState, PreTrainedModel

from ...Visualizers import AbstractVisualizer


class Controller(AbstractVisualizer):
    """
    A class that controls the training to increases the training's efficiency.
    """

    def __init__(self, minimum_loss: int|float, maximum_loss: int|float):
        """
        Initializes the controller object.

        :param minimum_loss: The minimum loss to reach, a goal to achieve.
        :param maximum_loss: The maximum loss to reach, a red flag.
        """

        if not all(isinstance(i, (int, float)) or i < 0.25 for i in [minimum_loss, maximum_loss]):
            raise ValueError("minimum_loss and maximum_loss must be integers or floats above 0.25.")

        self.increase_lr = False
        self.losses: List[float] = []
        self.last_increase = None
        self.condition: Callable[["Controller"], bool] = lambda slf: slf.losses[len(slf.losses) - 1] > slf.losses[len(slf.losses) - 11] - 1 and slf.losses[len(slf.losses) - 1] > minimum_loss
        self.stop_training_condition: Callable[[float], bool] = lambda loss: True if loss >= maximum_loss or loss <= minimum_loss else False

    def visualize_epoch(self, point: Literal["start", "end"],
                        current_epoch: int | float,
                        current_step: int,
                        trainer_state: TrainerState) -> dict:
        pass

    def visualize_step(self, point: Literal["start", "end"], current_epoch: int | float, current_step: int,
                       trainer_state: TrainerState) -> dict:
        pass

    def visualize_training(self, point: Literal["start", "end"], current_epoch: int | float, current_step: int,
                           trainer_state: TrainerState) -> dict:
        pass

    def visualize_optimizer(self, point: Literal["start", "end"], current_epoch: int | float, current_step: int,
                            current_lr: float, optimizer_hyperparams: List[
                Dict[str, float | Tuple[float, float] | List[Parameter]]],
                            model_parameters: Iterator[Parameter], trainer_state: TrainerState,
                            model: PreTrainedModel, optimizer: Optimizer, lr_scheduler: LambdaLR) -> dict:
        if point == "start":
            if self.increase_lr and (
                    self.last_increase is None or
                    self.last_increase < datetime.datetime.now(datetime.UTC)
                    - datetime.timedelta(minutes=10)
            ):
                current_lr += 1e-5
                self.last_increase = datetime.datetime.now(datetime.UTC)
            return {
                "new_lr": current_lr,
            }
        return {}

    def visualize_log(self, current_epoch: float | int,
                      current_step: int, current_loss: float,
                      logs: Dict[str, float | int]):
        if current_loss is not None:
            self.losses.append(current_loss)
        if len(self.losses) > 10:
            if self.condition(slf=self):
                self.losses.pop()
                self.increase_lr = True

        return {
            "stop": False if current_loss is None else self.stop_training_condition(current_loss)
        }


    def initialize(self, epochs: float, logging_steps: float, batch_size: int, learning_rate: float,
                   training_arguments: TrainingArguments):
        pass