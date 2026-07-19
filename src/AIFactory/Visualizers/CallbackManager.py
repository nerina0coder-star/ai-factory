import copy
from typing import Literal, Callable

import torch.optim.lr_scheduler
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, PreTrainedModel

from ..Visualizers.AbstractVisualizer import AbstractVisualizer


class CallbackManager(TrainerCallback):
    """
    A class that handles the managements of a visualizer.
    """

    def __init__(self):
        self.visualizers = []

    def add_visualizer(self, visualizer: AbstractVisualizer):
        if not isinstance(visualizer, AbstractVisualizer):
            raise TypeError(f"Expected an instance of AbstractVisualizer, given {type(visualizer).__name__}")
        self.visualizers.append(visualizer)

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        for visualizer in self.visualizers:
            visualizer.initialize(
                epochs=args.num_train_epochs,
                logging_steps=args.logging_steps,
                batch_size=args.per_device_train_batch_size,
                learning_rate=args.learning_rate,
                training_arguments=args
            )

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        for visualizer in self.visualizers:
            logs = kwargs["logs"]
            visualizer.visualize_log(
                state.epoch,
                state.global_step,
                logs.get("loss"),
                logs
            )

    def on_pre_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        for visualizer in self.visualizers:
            sched: torch.optim.lr_scheduler.LambdaLR = kwargs["lr_scheduler"]
            optim: torch.optim.Optimizer = kwargs["optimizer"]
            model: PreTrainedModel = kwargs["model"]
            output: dict = visualizer.visualize_optimizer(
                "start",
                current_epoch=state.epoch,
                current_step=state.global_step,
                current_lr=sched.get_last_lr()[0],
                optimizer_hyperparams=copy.deepcopy(optim.param_groups),
                model_parameters=model.parameters(),
                trainer_state=state,
                model=model,
                optimizer=optim,
                lr_scheduler=sched
            )

            CallbackManager.manage_update(output, control, optim)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.handle_normal(
            "visualize_epoch",
            "start",
            state,
            control
        )

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.handle_normal(
            "visualize_step",
            "start",
            state,
            control
        )

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.handle_normal(
            "visualize_training",
            "start",
            state,
            control
        )

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.handle_normal(
            "visualize_training",
            "end",
            state,
            control
        )

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.handle_normal(
            "visualize_step",
            "end",
            state,
            control
        )

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.handle_normal(
            "visualize_epoch",
            "end",
            state,
            control
        )

    def on_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        optim = kwargs["optimizer"]
        sched = kwargs["lr_scheduler"]
        model = kwargs["model"]
        for visualizer in self.visualizers:
            output = visualizer.visualize_optimizer(
                point="end",
                current_epoch=state.epoch,
                current_step=state.global_step,
                current_lr=optim.param_groups[0]["lr"],
                optimizer_hyperparams=copy.deepcopy(optim.param_groups),
                model_parameters=model.parameters(),
                trainer_state=state,
                model=model,
                optimizer=optim,
                lr_scheduler=sched
            )

            CallbackManager.manage_update(output, control)

    @staticmethod
    def manage_update(output: dict, control: TrainerControl, optim: torch.optim.Optimizer=None):
        if not isinstance(output, dict):
            return

        new_lr = output.get("new_lr")
        stop_epoch = output.get("stop_epoch")
        stop_training = output.get("stop")
        log = output.get("log")

        CallbackManager.update({
            "new_lr": new_lr,
            "should_stop_epoch": stop_epoch,
            "should_stop_training": stop_training,
            "should_log": log,
        }, control, optim)

    @staticmethod
    def update(dictionary: dict, control: TrainerControl, optim: torch.optim.Optimizer=None):
        if not isinstance(dictionary, dict):
            return

        for name, condition in dictionary.items():
            if not isinstance(condition, bool):
                continue

            if name == "new_lr" and optim is not None:
                optim.param_groups[0]["lr"] = condition
            else:
                setattr(control, name, condition)

    def handle_normal(self, method_name: str, point: Literal["start", "end"], state: TrainerState, control: TrainerControl):
        for vis in self.visualizers:
            calling: Callable[
                [Literal["start", "end"], float, float, TrainerState], dict
            ] = getattr(vis, method_name)
            out = calling(
                point,
                state.epoch,
                state.global_step,
                state
            )

            CallbackManager.manage_update(out, control)