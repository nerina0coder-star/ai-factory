from os import environ
from shutil import rmtree
from pathlib import Path
from typing import List

from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from rich import print
from torch import set_num_threads, set_num_interop_threads
from transformers import DataCollatorForLanguageModeling, DataCollator, TrainingArguments, \
    Trainer as TransformersTrainer, PreTrainedModel

from .Config import Config
from ..Helpers import TrainerHelper
from ..Visualizers import AbstractVisualizer, CallbackManager


class Trainer:
    def __init__(self):
        """
        Initialize the Trainer class.
        """
        self.__model_name__ = None
        self.__model__: PeftModel|PreTrainedModel|None = None
        self.__trained__ = None
        self.__config__: Config|None = None
        self.__lora_config__: LoraConfig|None = None
        self.__datacollator__: DataCollator|None = None
        self.__dataset__: Dataset|None = None
        self.__training_arguments__: TrainingArguments|None = None

    def set_config(self, config: Config):
        """
        Sets a config for the trainer.
        :param config: a Config object.
        """
        if not isinstance(config, Config):
            raise TypeError("config must be a Config object")
        self.__config__ = config
        self.__model__ = config._model

    def load_adapter(self, path: str):
        if isinstance(self.__model__, PeftModel):
            return False

        if isinstance(path, str):
            self.__model__ = PeftModel.from_pretrained(self.__model__, path)
            TrainerHelper.unfreeze_parameters(self.__model__)
            return True
        return False

    def prepare(self,
              path_to_parquet: str | List[str],
              loraconfig: LoraConfig = None,
              datacollator: DataCollator = None,
              training_arguments: TrainingArguments = None,
              dataset_label: str = "label",
              max_threads: int = None,
              log_every: int = 10
    ):
        """
        Prepares the object for the training.

        :param path_to_parquet: the path to our parquet file.
        :param loraconfig: a custom LoraConfig to set,
        visit https://huggingface.co/docs/peft/en/package_reference/lora#peft.LoraConfig
         to gather more information.
        :param datacollator: a DataCollator object from transformers.
        :param training_arguments: a TrainingArguments object from transformers.
        :param dataset_label: The label chosen for the dataset. If you didn't set it, leave it to be.
        :param max_threads: The maximum count of threads to use
        :param log_every: How often to log.
        """
        # Validating User input

        checking = {
            "path_to_parquet": [path_to_parquet, (list, str), "a list of strings or string"],
            "loraconfig": [loraconfig, (LoraConfig, None), "LoraConfig"],
            "datacollator": [datacollator, (DataCollator, None), "DataCollator"],
            "training_arguments": [training_arguments, (TrainingArguments, None), "TrainingArguments"],
            "dataset_label": [dataset_label, (str,), "string"],
            "max_threads": [max_threads, (int, None), "a positive integer", lambda: max_threads > 0],
            "log_every": [log_every, (int,), "A positive integer", lambda: log_every > 0]
        }

        for name, values in checking.items():
            if None in values[1] and values[0] is None:
                continue

            if None in values[1]:
                values[1] = list(values[1])
                values[1].remove(None)
                values[1] = tuple(values[1])

            base_check = lambda: values[0] is not None and isinstance(values[0], values[1])
            check = base_check
            if len(values) == 4:
                check = lambda: base_check() and values[3]()
            if not check():
                raise TypeError(f"{name} must be {values[2]}, got {type(values[0]).__name__} instead.")
        if not all(isinstance(path, str) for path in path_to_parquet):
            raise ValueError("All elements of path_to_parquet MUST be string.")

        # Preparing for training

        # Limiting threads

        threads = max_threads
        t = str(max_threads)

        try:
            environ["OMP_NUM_THREADS"] = t
            environ["MKL_NUM_THREADS"] = t
            environ["OPENBLAS_NUM_THREADS"] = t
            environ["VECLIB_MAXIMUM_THREADS"] = t
            environ["NUMEXPR_NUM_THREADS"] = t

            set_num_threads(threads)
            set_num_interop_threads(threads)
        except RuntimeError as _:
            pass
        del threads, t

        # Creating LoRA config with try and except block
        # to see if self.__config__ has been tempered with.

        try:
            if loraconfig is None:
                loraconfig = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    lora_alpha=64,
                    lora_dropout=0.1,
                    r=self.__config__._rank,
                    target_modules=self.__config__._attn_layers,
                    fan_in_fan_out=self.__config__._fifo,
                )
        except Exception as _:
            raise RuntimeError("An Exception acquired, make sure you have called Trainer.set_config() before.")

        # Creating others

        if datacollator is None:
            datacollator = DataCollatorForLanguageModeling(
                mlm=False,
                pad_to_multiple_of=None,
                tokenizer=self.__config__._tokenizer,
            )

        cpu_only = self.__config__._cpu_only

        dataset = load_dataset("parquet", data_files=path_to_parquet)

        def tokenize(txt):
            return self.__config__._tokenizer(txt[dataset_label], truncation=False, padding=True, return_tensors="pt")

        dtset = dataset.map(tokenize, batched=True, remove_columns=[dataset_label])

        if not training_arguments:
            training_arguments = TrainingArguments(
                logging_steps=10,
                warmup_steps=40,
                gradient_accumulation_steps=4,
                #---
                num_train_epochs=self.__config__.epochs,
                per_device_train_batch_size=self.__config__.batch_size,
                learning_rate=self.__config__._lr,
                #---
                fp16=not cpu_only,
                dataloader_pin_memory=not cpu_only,
                report_to="none",
                save_strategy="no"
            )

        self.__training_arguments__ = training_arguments
        self.__datacollator__ = datacollator
        self.__dataset__ = dtset["train"]
        self.__lora_config__ = loraconfig

    def train(self,
              visualizers: List[type[AbstractVisualizer]]|None = None,
              visualizers_kwargs: List[dict]|None = None,
              adapter_name: str|None = None,
              stop_when_interrupted: bool = True):
        """
        Trains the model.

        :param visualizers: The visualizers to be given to visualize
        and/or control the training.
        :param visualizers_kwargs: The kwargs to be given to the given visualizers.
        :param adapter_name: The name of the adapter to be made. Can be None.
        :param stop_when_interrupted: Whether to stop when user interrupts the training.
        """
        if not isinstance(stop_when_interrupted, bool):
            raise TypeError("stop_when_interrupted must be bool.")
        if visualizers is not None:
            if not isinstance(visualizers, list) or not isinstance(visualizers_kwargs, list):
                raise TypeError("both visualizers and visualizers_kwargs must be a list")
            if not all(issubclass(vizualizer, AbstractVisualizer) for vizualizer in visualizers):
                raise TypeError("all visualizers must be children of AbstractVisualizer")
            if not all(isinstance(kwrg, dict) for kwrg in visualizers_kwargs):
                raise TypeError("visualizers_kwargs must be a list of dicts")
            if not len(visualizers) == len(visualizers_kwargs):
                raise ValueError("visualizers and visualizers_kwargs must have the same length")
        if adapter_name is not None:
            if not isinstance(adapter_name, str):
                raise TypeError("adapter_name must be a string")

        callback = CallbackManager()
        if visualizers:
            for index in range(len(visualizers)):
                callback.add_visualizer(visualizers[index](**visualizers_kwargs[index]))

        training_model = self.__model__
        if not isinstance(self.__model__, PeftModel):
            try:
                training_model = get_peft_model(
                    self.__model__,
                    self.__lora_config__,
                    adapter_name="ai_factory_adapter" if adapter_name is None else adapter_name
                )
            except Exception as _:
                raise RuntimeError("An error acquired when applying PEFT, make sure to call Trainer.prepare() beforehand.")

        self.__trained__ = False

        if getattr(training_model.config, "loss_type", None) is None:
            training_model.config.loss_type = "ForCausalLMLoss"

        try:

            trainer = TransformersTrainer(
                data_collator=self.__datacollator__,
                train_dataset=self.__dataset__,
                model=training_model,
                args=self.__training_arguments__,
                callbacks=[callback]
            )

            trainer.train()
            self.__trained__ = True
            self.__model__ = training_model
            self.__model_name__ = adapter_name
            return

        except KeyboardInterrupt:
            print(f"[bold yellow]Training interrupted by user, stopping the training{"" if stop_when_interrupted else " and proceeding"}...[/]")
            if stop_when_interrupted:
                raise RuntimeError("Training Stopped")
        except Exception as e:
            raise RuntimeError(f"Something went wrong while training the model. {e}")

    def save_model(self, path: str, f: bool|None = None) -> bool:
        """
        Saves the model.

        Note, the saved model isn't a PeftModel, but rather a pretrained model.

        :param path: The path to save the model to.
        :param f: Whether to force-save the model, any pre-existing data would be overwritten.
        :return: A True if succeeded, False otherwise.
        """
        if not isinstance(path, str):
            return False
        pathed = Path(path + self.__model_name__)
        if pathed.exists():
            if not f == True:
                raise FileExistsError(f"{path} already exists")
            rmtree(path)

        try:
            if self.__trained__:
                model = self.__model__
                tokenizer = self.__config__._tokenizer
                if isinstance(model, PeftModel):
                    model = model.merge_and_unload()
                TrainerHelper.remove_prefix("base_model.model.", model)
                TrainerHelper.remove_peft_leftovers(model)
                tokenizer.save_pretrained(path)
                model.save_pretrained(path, save_peft_format=False)
                return True
        except Exception as e:
            raise RuntimeError(f"Something went wrong while saving the model, \
            make sure you called the Trainer.train() function beforehand. {e}")

        return False

    def save_adapter(self, path: str, f: bool|None = None) -> bool:
        """
        Saves the adapter trained.

        :param path: The path to save the adapter to.
        :param f: whether to force-save or not. Any pre-existing data would be overwritten.
        :return: A True if succeeded, False otherwise.
        """
        if not isinstance(path, str):
            return False
        if Path(path + self.__model_name__).exists():
            if not f == True:
                raise FileExistsError(f"{path} already exists")
            rmtree(path)
        if not hasattr(self, "__model__") or not isinstance(self.__model__, PeftModel):
            raise RuntimeError("The model is not a PeftModel, make sure to train first.")

        pathed = Path(path)
        pathed.parent.mkdir(parents=True, exist_ok=True)

        try:
            model = self.__model__
            model.save_pretrained(path)
            return True
        except Exception as e:
            raise RuntimeError(f"Something went wrong while saving the adapter. {e}")
