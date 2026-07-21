from abc import ABC, abstractmethod
from typing import Literal, List, Tuple, Dict, Iterator

from torch.nn import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import TrainingArguments, TrainerState, PreTrainedModel


class AbstractVisualizer(ABC):

    @abstractmethod
    def initialize(self,
                   epochs: float,
                   logging_steps: float,
                   batch_size: int,
                   learning_rate: float,
                   training_arguments: TrainingArguments):
                """
                Used to receive data after the trainer object is initialized.

                :param epochs: The number of epochs the training will long.
                :param logging_steps: After how many steps will it log to the user.
                :param batch_size: The batch size for training.
                :param learning_rate: The starting learning rate defined by the user.
                :param training_arguments: The training arguments passed to the trainer object.
                """

    @abstractmethod
    def visualize_epoch(self,
                        point: Literal["start", "end"],
                        current_epoch: int|float,
                        current_step: int,
                        trainer_state: TrainerState) -> dict:
                """
                Used to receive data when an epoch starts and ends.

                :param point: The point where this is called at.
                :param current_epoch: The current epoch precise to decimals.
                :param current_step: The current step.
                :param trainer_state: The trainer state, containing live data of the training.
                :return: A dictionary containing the following keys(you can return None or a dict of None too):
                stop - Stops the whole training.
                stop_epoch - Stops the running epoch.
                log - Logs as soon as possible after this.
                """

    @abstractmethod
    def visualize_step(self,
                       point: Literal["start", "end"],
                       current_epoch: int|float,
                       current_step: int,
                       trainer_state: TrainerState) -> dict:

            """
            Used to receive data when a step starts and ends.

            :param point: The point where this is called at.
            :param current_epoch: The current epoch precise to decimals.
            :param current_step: The current step.
            :param trainer_state: The trainer state, containing live data of the training.
            :return: A dictionary containing the following keys(you can return None or a dict of None too):
            stop - Stops the whole training.
            stop_epoch - Stops the running epoch.
            log - Logs as soon as possible after this.
            """

    @abstractmethod
    def visualize_training(self,
                           point: Literal["start", "end"],
                           current_epoch: int|float,
                           current_step: int,
                           trainer_state: TrainerState) -> dict:
        """
        Used to receive data when training starts and ends.

        :param point: The point where this is called at.
        :param current_epoch: The current epoch precise to decimals.
        :param current_step: The current step.
        :param trainer_state: The trainer state, containing live data of the training.
        :return: A dictionary containing the following keys(you can return None or a dict of None too):
        stop - Stops the whole training.
        stop_epoch - Stops the running epoch.
        log - Logs as soon as possible after this.
        """

    @abstractmethod
    def visualize_optimizer(self,
                            point: Literal["start", "end"],
                            current_epoch: int|float,
                            current_step: int,
                            current_lr: float,
                            optimizer_hyperparams: List[Dict[str, float|Tuple[float, float]|List[Parameter]]],
                            model_parameters: Iterator[Parameter],
                            trainer_state: TrainerState,
                            model: PreTrainedModel,
                            optimizer: Optimizer,
                            lr_scheduler: LambdaLR) -> dict:
        """
        The only method you can gain access to lr_scheduler, optimizer and model in.
        Used to gain logical control in training and visualize needed data.

        :param point: The point where this is called at.
        You can tweak lr when this is set to start by returning a
        dictionary containing new_lr.
        :param current_epoch: The current epoch precise to decimals.
        :param current_step: The current step.
        :param current_lr: The current learning rate.
        :param optimizer_hyperparams: hyperparameters of the optimizer,
        you can change any data as they are not real, except for the params item of this list.
        This dictionary contains necessary data such as 'params'(torch.nn.Parameter), 'lr'(float), 'weight_decay'(float), 'betas'(a tuple containing two floats), 'eps'(float).
        The structure of this can be described as a list containing 2 dictionaries that contain the keys named above inside with the exact same data type.
        To modify the hyperparameters dynamically, please use the advanced option "optimizer"
        :param model_parameters: The parameters of the model.
        :param trainer_state: The trainer state, containing live data of the training.
        :param model: (Advanced) The model in Training.
        :param optimizer: (Advanced) The optimizer.
        :param lr_scheduler: (Advanced) the lr_scheduler.
        :return: A dictionary containing the following keys(you can return None or a dict of None too):
        new_lr - Dynamically change the lr for the next step.
        stop - Stops the whole training.
        stop_epoch - Stops the running epoch.
        log - Logs as soon as possible after this.
        """

    @abstractmethod
    def visualize_log(self,
                      current_epoch: float|int,
                      current_step: int,
                      current_loss: float,
                      logs: Dict[str, float|int]):
        """
        Uses the log event to visualize loss, lr, current step, current epoch or alike.

        :param current_epoch: The current epoch precise to decimals.
        :param current_step: The current step.
        :param current_loss: The current loss.
        :param logs: The complete log dictionary. Use carefully. Contains other data such as lr.
        """
