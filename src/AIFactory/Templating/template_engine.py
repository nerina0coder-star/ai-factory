import threading
from pathlib import Path
from typing import Literal, List, Callable, Dict, Any
from warnings import warn

import pandas
import re2
import numpy as np
from numpy import dtype, ndarray

from ..Helpers import FixingHelper
from ..Templating.concepts import Concepts


class TemplateEngine:

    def __init__(self, concepts: Concepts, style: Literal["Q&A", "CAUSAL"], **kwargs):
        """
        Initialize a template engine.
        It would be kind of you to set
        style only to Q&A or CAUSAL unless you want this to break.

        set chat_template and roles in kwargs if you choose Q&A.
        roles should be a list that contains roles. e.g.:

        ["user", "assistant"]

        And chat_template should be the apply_chat_template function of transformers AutoTokenizer class
        or a custom function that accepts a list of dicts that contain the keys:

        role, content.

        alongside two booleans: tokenize and add_generation_prompt.

        :param concepts: must be a child of `Concepts`
        :param style: must be Q&A or CAUSAL

        :raises TypeError: concepts must be a child of Concepts.
        :raises RuntimeError: An error acquired while testing chat_template: {details}
        :raises Etc...:
        """
        if not isinstance(concepts, Concepts):
            raise TypeError("concepts must be a child of Concepts")
        if style not in ["Q&A", "CAUSAL"]:
            raise ValueError("style must be Q&A or CAUSAL")
        if style == "Q&A" and \
                (kwargs.get("chat_template") is None or kwargs.get("roles") is None):
            raise ValueError("chat_template and roles must be set for Q&A")
        if style == "Q&A" and \
                not callable(kwargs["chat_template"]) and \
                not isinstance(kwargs["roles"], list):
            raise TypeError("chat_template must be a callable and roles must be a list")

        if style == "Q&A":
            try:
                assert isinstance(
                    kwargs['chat_template']([{"role": "test", "content": "test"}], add_generation_prompt=False,
                                            tokenize=False), str)
            except Exception as e:
                raise RuntimeError("An exception acquired while testing chat_template: ", str(e))

        # Creating problem detection lambdas.
        lambdas = []
        descriptions = []

        lambdas.append(lambda x: x.count("|") > 1)
        descriptions.append(
            """
            You have more than 1 pipes in one of your templates, be aware that the part before the 1st pipe is \
            the function and the part afterward are the functions.
            """
        )

        lambdas.append(lambda x: (x.count("}") + x.count("{")) / 2 > 1)
        descriptions.append(
            """
            You have more than 1 closing/opening curly brace in your template. \
            Make sure to escape inner curly braces. Both opening and closing curly braces.
            """
        )

        lambdas.append(lambda x: x.count("${") > 1)
        descriptions.append(
            """
            You have more than 1 call opener, make sure that's meant to be
            a real one! If not, use the function ${_} to get a string like ${.
            """
        )

        # Saving

        self.style = style
        self.chat_template = kwargs.get("chat_template")
        self.roles = kwargs.get("roles")
        self.__functions__ = concepts.__map__
        self.__zones__: List[int] = []
        self.__last_evaluated__: List[np.typing.NDArray] = []
        self.__packs__: List[dict] = []
        self.__helper__: FixingHelper = FixingHelper(
            lambdas,
            descriptions
        )


    def register_pack(self,
                      pack_name: str,
                      pack_importance: int,
                      pack_index: int|Literal["highest", "lowest", "middle"]) -> int:
        """
        Registers a pack. Packs are groups of templates used to organize the dataset even more.
        Each pack is evaluated by its index. The lower the index, the faster it gets evaluated.
        :param pack_name: The name of the pack.
        :param pack_importance: The importance of the pack. This will be used when evaluating
        the templates as it increases the frequency of its child templates in the output.
        :param pack_index: The index of the pack. The lower the index, the faster it gets evaluated.
        :return: The count of all packs.
        """

        # does the validation
        if not isinstance(pack_name, str):
            raise TypeError(f"pack_name must be string, not {type(pack_name).__name__}")
        if not isinstance(pack_importance, int):
            raise TypeError(f"pack_importance must be integer, not {type(pack_importance).__name__}")
        if not isinstance(pack_index, (str, int)):
            raise TypeError(f"pack_index must be integer or string, not {type(pack_index).__name__}")

        if isinstance(pack_index, str) and pack_index not in ["highest", "lowest", "middle"]:
            raise ValueError(f"pack_index must be highest, lowest, or middle. Got {pack_index}")

        if list(filter(lambda x: x.get("name") == pack_name, self.__packs__)):
            raise ValueError(f"pack_name must be unique.")

        # Changes the string to int
        base_call = lambda index, x: self.__packs__.insert(index, x)

        if pack_index == "highest":
            call = lambda x: self.__packs__.append(x)
        elif pack_index == "lowest":
            call = lambda x: base_call(len(self.__packs__) // 2, x)
        elif pack_index == "middle":
            call = lambda x: base_call(0, x)
        else:
            call = lambda x: base_call(pack_index, x)

        # Does the appending
        appending = {
            "name": pack_name,
            "importance": pack_importance,
            "templates": []
        }
        call(appending)
        return len(self.__packs__)

    def register_template(self,
                          pack_name: str,
                          qa: list | None = None,
                          template: str | None = None,
                          importance: int = None,
                          zone_start: bool = False,
                          zone_end: bool = False,):
        """
        Registers a template and gives it an importance.
        Note that the template can only contain keyword-arguments. Here are a few examples:

        Once upon a time, the time was ${integer} -> integer would be called with no arguments.

        Once upon a time, it was ${time|start=10, end=20} # note the spacing -> time would be called,
        and it would receive an argument named start with value "10"(string) and another named end with value "20"(string).

        :param pack_name: The name of the pack this belongs to
        :param importance: the importance of this template.
        :param zone_start: flags the start of an importance zone.
        :param zone_end: flags the end of an importance zone.
        :param qa: (set if style == Q&A) Must be a list, every item in this is used by the role at the same index as the item.
        :param template: (set if style == CAUSAL) Must be a string.
        """
        # Handing validation
        if importance is None:
            importance = 1
            for i in self.__zones__:
                importance *= i

        if not self.__zones__ and zone_end:
            raise RuntimeError("Cannot end a zone that's not started")
        if not all(item is not None for item in [importance, zone_start, zone_end]):
            raise TypeError("importance, zone_start and zone_end must be set")
        if qa is None and template is None:
            raise TypeError("either qa or template must be set")
        if qa is None and self.style == "Q&A":
            raise TypeError("qa must be set for Q&A")
        if template is None and self.style == "CAUSAL":
            raise TypeError("template must be set for CAUSAL")
        if not isinstance(pack_name, str):
            raise TypeError("pack_name must be string")
        if not list(filter(lambda x: x.get("name") == pack_name, self.__packs__)):
            raise ValueError(f"pack must be defined. Got unknown pack_name {pack_name}")
        # Handling importance
        if zone_start:
            self.__zones__.append(importance)
        if zone_end:
            self.__zones__.pop()

        # Registering the template
        registering = template
        if self.style == "Q&A":
            items = []
            for i in range(len(qa)):
                item = {}
                item.setdefault("role", self.roles[i])
                item.setdefault("content", qa[i])
            registering = self.chat_template(
                items,
                add_generation_prompt=False,
                tokenize=False
            )

        assert registering is not None

        concept_usage = re2.compile(r"\$\{.*?[^\\]\}")
        param_usage = re2.compile(r"[a-zA-Z0-9_]*?\s*[^\\]=\s*[^,]*")
        get_kname = re2.compile(r"[a-zA-Z0-9_]*?[^\\]=")
        get_kval = re2.compile(r"[^\\]=\s*[^,]*")  # Must use kwargs and not args, and can't put , in there.
        usages = concept_usage.findall(registering)

        used = {"fulls": [], "concepts": [], "params": {}, "importance": importance}
        for usage in usages:
            usage = str(usage)
            # Extracting calls and params from usages.
            # Extracting concepts
            index = usage.find('|')
            item = usage[2:-1]
            params = None
            if index != -1:
                item = usage[2:index]
                params = usage[index:-1]


            used['concepts'].append(item)
            used['params'].setdefault(item, {})
            used["fulls"].append(usage)
            if params is not None:
                params = params \
                    .replace(r"\}", "}") \
                    .replace(r"\{", "{")

                used_params = list(str(param) for param in param_usage.findall(params))
                for par in used_params:
                    par = str(par)
                    kname = get_kname.match(par).group()[:-1]
                    kval = get_kval.search(par).group()[2:].replace(r"\,", ",").replace(r"\=", "=")
                    used['params'].setdefault(item, {}).setdefault({kname: kval})
        template = [registering, used]
        pack: dict = list(filter(lambda x: x.get("name") == pack_name, self.__packs__))[0]
        pack["templates"].append(template)

    def eval(self) -> tuple[ndarray[tuple[Any, ...], dtype[Any]], FixingHelper]:
        """
        Evaluates the templates registered by register_template.
        Be careful to not set importances to high numbers, because the formula used is:

        template importance * pack importance

        :returns: A list of all evaluated templates.
        """
        packs = self.__packs__
        funcs = self.__functions__

        out: np.typing.NDArray[np.str_] = np.array([])
        last: list = []

        threads = []
        indexed_results = {
        }

        for pack in range(len(packs)):
            p = packs[pack]

            indexed_results[pack] = []

            t = threading.Thread(
                target=self.__beval__,
                kwargs={
                    "pack": p,
                    "save_to": indexed_results[pack], # TODO Use multiprocessing for CPU-bound call
                    "funcs": funcs                    # TODO Later
                }
            )
            threads.append(t)

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        for sublast, subout in list(indexed_results.values()):
            last.extend(sublast)

            out = np.concatenate([out, subout], dtype=str)

        self.__helper__.last_thing = last

        self.__check__(out, mx=100000)

        self.__last_evaluated__.append(out)

        self.__check__()

        return self.__last_evaluated__[-1], self.__helper__

    def merge_onto(self, other: "TemplateEngine") -> "TemplateEngine":
        self.__last_evaluated__.extend(other.__last_evaluated__)
        self.__check__()
        return self

    def save_to_parquet(self,
                        custom_path: str,
                        evaluated: list | None = None,
                        custom_label: str | None = "label",
    ) -> pandas.DataFrame:
        """
        Saves the dataset to a parquet file.

        :param custom_path: A custom path to the file.
        :param evaluated: The evaluated items as a list. By default, it's the last evaluated item.
        :param custom_label: A custom label for the dataset.
        :returns: The dataset as a pandas DataFrame.
        """
        # Validating user input
        if evaluated is None:
            evaluated = self.__last_evaluated__[-1]
        if custom_label is None:
            custom_label = self.style

        if not isinstance(evaluated, (list, np.ndarray)):
            raise TypeError("evaluated must be a list")
        if not isinstance(custom_label, str):
            raise TypeError("custom_label must be a string")
        if not isinstance(custom_path, str):
            raise TypeError("custom_path must be a string")
        if not all(isinstance(i, str) for i in custom_path):
            raise TypeError("all elements of custom_path must be string")

        # Warnings
        if len(evaluated) < 100: warn("The number of evaluated items is too low, the model may not learn the pattern.")

        # saving

        path = Path(custom_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        to_save = {custom_label: evaluated}
        dtframe = pandas.DataFrame(to_save)

        dtframe.to_parquet(custom_path,
                           engine="pyarrow",
                           compression="zstd",
                           compression_level=3) # 3 already is enough as the
        # importance makes too much patterns in the dataset,
        # Causing zstd to compress it phenomenally well.

        return dtframe

    def __beval__(self, pack: Dict[str, List[dict]], save_to: List, funcs: List[Callable]):
        """
        Used to save evaluate packs in threads.

        :param pack: The pack to use
        :param save_to: The list to save to
        :param funcs: The functions available
        """
        templates = pack["templates"]
        last = []
        out: np.typing.NDArray[np.str_] = np.array([])
        for template in templates:
            t = template[1]

            complete = template[0]

            fulls = t["fulls"]
            concepts = t["concepts"]
            params = t["params"]
            importance = t["importance"]

            last.append(complete)

            for _ in range(importance * pack["importance"]):
                adding = ""

                for index in range(len(concepts)):
                    concept = concepts[index]
                    param = params[concept]
                    full = fulls[index]

                    evaluated = funcs[concept](**param)
                    adding = complete.replace(full, evaluated)

                out = np.append(out, [adding])

        for i in [last, out]:
            save_to.append(i)

    def __check__(self, explicit=None, mx=None):
        if explicit is None:
            explicit = self.__last_evaluated__
        if mx is None:
            mx = 1000
        cond = lambda: len(explicit) > mx
        if cond():
            while cond():
                explicit.pop(0)
            warn("Erased some data to prevent memory overload."
                 "Please don't put so much packs/templates in the engine.")