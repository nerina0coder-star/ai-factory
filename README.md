![Author Mail](https://img.shields.io/badge/Email-github.py.coder%40gmail.com.-red?logo=github)
![Version](https://img.shields.io/badge/Version-v0.0.1-orange?logo=github)
![License](https://img.shields.io/badge/License-MIT-red?logo=github)
![Usablity](https://img.shields.io/badge/Usablity-Low-orange)

# 1. What is this?
## In Short Terms
A tool that simplifies the process of training an AI
## Completely Explained
A tool that uses the 🤗 Transformers Library and pytorch to train an AI behind the scene,
making it easier for beginners to enter the world of AI fine-tuning.
### Packages
1. AIFactory.Template
- Contains classes for creating datasets, such as Concepts and TemplateEngine
2. AIFactory.Helpers
- Contains classes for helping the user, such as FixingHelper and TrainerHelper
3. AIFactory.Training
- Contains classes for training an AI, such as Config and Trainer
4. AIFactory.Visualizers
- Contains classes for visualizing/controlling the process of training, such as AbstractVisualizer.
5. AIFactory.Visualizers.BuiltinVisualizers
- Contains existing classes visualizing, such as Controller and Seaplot. It can be accessed
via AIFactory.BuiltinVisualizers too.
### Full Example
#### Creating the templates
```python
import json
from AIFactory.Templating import Concepts, TemplateEngine
from AIFactory.Helpers import FixingHelper
from numpy.typing import NDArray
# Creating concept class
class MyConcepts(Concepts):
    """
    My Own Custom Concepts
    """

    def __init__(self):
        self.__map = []

        super().__init__(function_map=self.__map)

# Creating dataset
engine = TemplateEngine(MyConcepts(), "CAUSAL")
engine.register_pack("my_pack", 10, "lowest")  # creates a pack to organize the data.
_ = engine.register_template
_("Once upon a time, the time was ${integer}, and what I'm saying is ${boolean}. \
I say ${history|adding_to_history=clerk} and last time I said ${history|index_behind_newest=1}"
  )# Note the spacing, No space before or after =. Escape =s and {}s via \. 
dt_and_helper: tuple[NDArray, FixingHelper] = engine.eval()
dataset = dt_and_helper[0]
helper = dt_and_helper[1]

print(json.dumps(dataset, indent=2))
print("Is this helpful?")
inp = input()[0].lower()

if not inp == "y":
    helper.this_thing_is_garbage()
    
engine.save_to_parquet("./path/to/.parquet")

del dt_and_helper, engine, inp, dataset, helper
```
#### Training a model
```python
from AIFactory.Training import Config, Trainer
from AIFactory import BuiltinVisualizers
from numpy.typing import NDArray
from numpy import dtype
from typing import Any

# Training

visualizers = [
    BuiltinVisualizers.Controller,
#    BuiltinVisualizers.Seaplot(update_every=10) In Development
]
kwargs = [
    {
        "minimum_loss": 0.5,
        "maximum_loss": 10
    },
#    {
#        "update_every": 10
#    } as said, in development
]

config = Config(
    epochs=1,
    model_path="./path/to/old_model"
)

trainer = Trainer()

trainer.set_config(config=config)
trainer.prepare(
    "./path/to/.parquet"
)
trainer.train(visualizers=visualizers, visualizers_kwargs=kwargs)

trainer.save_model("./path/to/model") # or trainer.save_adapter("./path/to/adapter"). Whichever you prefer.

del config, trainer, visualizers
```
#### Fixing a model
```python
from AIFactory.Helpers import FixingHelper, TrainerHelper
from transformers import AutoModelForCausalLM
from peft import PeftModel


model = AutoModelForCausalLM.from_pretrained("./path/to/existing_model")

TrainerHelper.remove_peft_leftovers(model)
TrainerHelper.remove_prefix("base_model.model.", model)

pft = PeftModel.from_pretrained(model, "./path/to/existing_adapter")

TrainerHelper.unfreeze_parameters(pft)

... # Other kind of things you wanna do.
```