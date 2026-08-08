import json
from collections.abc import Iterable
from os import PathLike
from pathlib import Path
from typing import List, Literal

from peft import prepare_model_for_kbit_training
from torch import float16, float32, float64
from transformers import AutoModelForCausalLM, AutoTokenizer, TokenizersBackend, SentencePieceBackend, PreTrainedModel, \
    AutoConfig, PreTrainedConfig, BitsAndBytesConfig
from rich import print

class Config:
    def __init__(self,
                 epochs: int = 1,
                 attention_layers: List[str] = None,
                 max_layers: int = 1,
                 model_path: str = None,
                 rank: int = 4,
                 precision: Literal["16", "32", "64"] = "32",
                 fan_in_fan_out: bool = False,
                 cpu_only: bool = False,
                 batch_size: int = 1,
                 learning_rate: float|Literal["low", "medium", "high", "extreme"] = "medium",
                 max_memory: int = 6,
                 cnf: AutoConfig = None,
                 tkn: AutoTokenizer = None,
                 mdl: PreTrainedModel = None,
                 bnb: BitsAndBytesConfig = None):
        """
        Creates a Config object that contains basic information about the training.
        :param epochs: The number of epochs the training will last.
        :param attention_layers: The attention layers of the LM model.
        :param max_layers: The maximum number of layers to include in the training.
        :param model_path: The path to the model.
        :param rank: The rank of the PEFT.
        The higher the rank, the more resource exhaustive the training.
        As the rank increases, the model's capacity of learning also increases.
        :param precision: The precision of the training. It must be a power of 2.
        People usually prefer 16 precision as it's perfect for LoRa.
        32 precision is also a good choice for the best of both worlds.
        64 precision is used in some cases as it's the best of high precisions(in sense of not being resource exhaustive).
        :param fan_in_fan_out: Some models need this to perform.
        :param batch_size: The batch size for training.
        :param max_memory: The maximum memory the AI can fill in GigaBytes.
        :param bnb: A bitsandbytes config for Quantized Low Rank Adaption(QLoRA).
        """
        # Turning string to lr
        lr_mapping = {
            "low": 8e-6,
            "medium": 5e-5,
            "high": 1e-4,
            "extreme": 5e-4
        }

        if isinstance(learning_rate, str):
            learning_rate = lr_mapping.get(learning_rate)
            if learning_rate is None:
                raise ValueError(f"Unknown option {learning_rate}")
        # Validating user input
        integer_pos = lambda x: x < 1

        checking = {
            "learning_rate": [learning_rate, float, "float or string"],
            "precision": [precision, str, "string, either 16, 32, or 64", lambda: precision not in ["16", "32", "64"]],
            "attention_layers": [attention_layers, [list, None], "list with string elements", lambda: attention_layers is not None and not all(isinstance(layer, str) for layer in attention_layers)],
            "model": [model_path, str, "string"],
            "fan_in_fan_out": [fan_in_fan_out, bool, "bool"],
            "cpu_only": [cpu_only, bool, "bool"],
        }

        for i, j in {
            "precision": precision,
            "rank": rank,
            "epochs": epochs,
            "max_layers": max_layers,
            "batch_size": batch_size,
            "max_memory": max_memory
        }.items():
            checking.setdefault(i, [j, int, "positive integer", lambda: integer_pos(j)])

        for name, values in checking.items():
            base_check = lambda: (values[0] is None and None not in values[1]) or (values[0] is not None and not isinstance(values[0], tuple(filter(lambda x: x is not None, values[1] if isinstance(values[1], Iterable) else [values[1]]))))
            check = base_check
            if len(values) == 4:
                check = lambda: base_check() or values[3]()
            if check():
                raise TypeError(f"{name} must be {values[2]}")

        model, conf, tokenizer = self.__st__(
            mdl if mdl is not None and isinstance(mdl, PreTrainedModel) else None,
            cnf if cnf is not None and isinstance(cnf, AutoConfig) else None,
            tkn if tkn is not None and isinstance(tkn, AutoTokenizer) else None,
            bnb if bnb is not None and isinstance(bnb, BitsAndBytesConfig) else None,
            model_path if model_path is not None and isinstance(model_path, str) else None,
            precision,
            cpu_only,
            max_memory,
        )


        if attention_layers is None:
            print("[bold yellow]Trying to find the model's attention layers...[/]")

            attention_layers = []

            # Finding the attention laters
            for name, module in model.named_modules():
                if not any(k in name for k in ["attn", "proj", "attention", "key", "value", "dense", "query"]):
                    continue
                for subname, submodule in module.named_modules():
                    if subname == "": # To skip the root module
                        continue
                    attention_layers.append(subname)

            # Catching the doubles
            attention_layers = list(set(attention_layers))

            # Confirming the layers to make sure no error is raised later
            print(f"[bold green]Found layers: {"".join(f"{layer} " for layer in attention_layers)}[/]"
                  "\n[black]Confirm? y/n[/]")
            answer = input()[0].lower()

            if answer != 'y':
                raise RuntimeError("Please confirm the model's attention layers or pass the attention layers explicitly.")

        self._all_layers = attention_layers
        self._attn_layers = attention_layers[:max_layers]
        self._model_path = model_path
        self._model: PreTrainedModel = model
        self._tokenizer: TokenizersBackend | SentencePieceBackend = tokenizer
        self.epochs = epochs
        self._rank = rank
        self._fifo = fan_in_fan_out     # Surely not first in first out
        self._lr: float = learning_rate
        self._cpu_only: bool = cpu_only
        self.batch_size = batch_size

    def save_config(self, path: PathLike|str,
                    model_name: str,
                    other_config_names: List[str] = None,
                    other_configs: List["Config"] = None):
        """
        Saves the current configurations of the Config file.

        :param path: The path to save the config at. E.g., /home/user/configs/my_config.
        The extension is automatically added.
        :param model_name: The name of the model.
        :param other_config_names: The name of the other Config objects.
        :param other_configs: Other Config objects to save alongside the current Config.
        """
        # Validating user input
        if other_configs is not None:
            if not isinstance(other_configs, list):
                raise TypeError("other_configs must be a list")
            if not all(isinstance(i, Config) for i in other_configs):
                raise ValueError("other_configs must only contain Config objects")
        if not isinstance(path, (PathLike, str)):
            raise TypeError("path must be a PathLike, str or Path")
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string")

        # Adding extension if not present and creating the file.

        if isinstance(path, str):
            path = Path(path + ".factory-conf.json"
                        if not path.endswith(".factory-conf.json")
                        else path)

        if not path.name.endswith(".factory-conf.json"):
            path.name += ".factory-conf.json"

        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

        # Writing the config.
        to_write = \
            [
                {
                    "name": model_name,
                    "all_layers": self._all_layers,
                    "model_path": self._model_path
                }
            ]
        for conf in range(len(other_configs)):
            to_write.append(
                {
                    "name": other_config_names[conf],
                    "all_layers": other_configs[conf]._all_layers,
                    "model_path": other_configs[conf]._model_path
                }
            )

        path.write_text(
            json.dumps(
                to_write,
                indent=4,
            )
        )

    @staticmethod
    def load_config(path_to_config: PathLike|str,
                    model_name: str,
                    epochs: int = 1,
                    max_layers: int = 1,
                    rank: int = 4,
                    precision: Literal["16", "32", "64"] = "32",
                    fan_in_fan_out: bool = False,
                    **kwargs):
        """
        Loads a config from a JSON file.

        :param path_to_config: The path to the config. Note that the extension will NOT be automatically added.
        :param model_name: The name of the model to load.
        :param epochs: See init method of Config class.
        :param max_layers: ...
        :param rank: ...
        :param precision: ...
        :param fan_in_fan_out: ...
        :param kwargs: Other keyword-arguments to give to the Config class.
        :return: A Config object.
        """
        # Turning into a path
        if isinstance(path_to_config, str):
            path_to_config = Path(path_to_config + ".factory-conf.json"
                                  if not path_to_config.endswith(".factory-conf.json")
                                  else path_to_config)

        # Validating user input
        if not isinstance(path_to_config, (PathLike, str)):
            raise TypeError("path_to_config must be a PathLike, str or Path")

        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string")

        if not path_to_config.exists():
            raise FileNotFoundError(path_to_config)

        # Loading config
        text = path_to_config.read_text()
        jsonified = json.loads(text)

        config = list(filter(lambda i: i["name"] == model_name, jsonified))[0]

        layers = config["all_layers"]
        path = config["model_path"]

        return Config\
            (
                epochs=epochs,
                attention_layers=layers,
                max_layers=max_layers,
                model_path=path,
                rank=rank,
                precision=precision,
                fan_in_fan_out=fan_in_fan_out
            )

    def __st__(self, mdl: PreTrainedModel = None,
               cnf: PreTrainedConfig = None,
               tkn: TokenizersBackend|SentencePieceBackend = None,
               bnb: BitsAndBytesConfig = None,
               mdl_path: str = None,
               precision: str = None,
               cpu_only: bool = None,
               mm: int = None) -> tuple[PreTrainedModel,
    PreTrainedConfig,
    TokenizersBackend|SentencePieceBackend]:
        """
        Returns a model, config, and tokenizer.

        :param mdl: a model
        :param cnf: a config
        :param tkn: a tokenizer
        :param mdl_path: path to model
        :param precision: precision
        :param cpu_only: whether to work CPU-only
        :param mm: max memory
        :param bnb: A bits and bytes config.
        :return: a tuple of model, config, and tokenizer
        """
        tokenizer = tkn
        conf = cnf
        model = mdl

        # Setting configs
        float_dict = {
            "float16": float16,
            "float32": float32,
            "float64": float64
        }

        Path("./.offloader").mkdir(parents=True, exist_ok=True)

        if tokenizer is None or not isinstance(tokenizer, (TokenizersBackend, SentencePieceBackend)):
            tokenizer: TokenizersBackend|SentencePieceBackend = \
                AutoTokenizer.from_pretrained(mdl_path)

        if conf is None or not isinstance(conf, PreTrainedConfig):
            conf: PreTrainedConfig = AutoConfig.from_pretrained(mdl_path)

        if model is None or not isinstance(model, PreTrainedModel):
            model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                mdl_path,
                dtype=float_dict.get(f"float{precision}"),
                device_map="auto" if not cpu_only else None,
                offload_folder="./.offloader",
                max_memory={0: f"{mm}GB"},
                config=conf,
                quantization_config=bnb
            )

            if bnb is not None:
                model = prepare_model_for_kbit_training(model)


        return model, conf, tokenizer
