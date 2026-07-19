from peft import PeftModel
from transformers import PreTrainedModel


class TrainerHelper:
    @staticmethod
    def remove_prefix(prefix: str, model: PreTrainedModel):
        """
        Used when the model has MISSING/UNEXPECTED errors related to a prefix from the internal structure.
        This function removes the prefix from the state dict and fixes the issue.

        The error is because after training the model, the internal structure of the model,
        changed by the PEFT, was not changed.

        If you see a prefix like "base_model.model.", please include the dot.

        :param prefix: The prefix.
        :param model: The model's name.
        """
        for k, v in model.state_dict().items():
            if k.startswith(prefix):
                k = k[len(prefix):]
                model.state_dict()[k] = v

    @staticmethod
    def unfreeze_parameters(model: PeftModel):
        """
        Unfreezes LoRA weights, required for re-training an adapter.

        :param model: The model with the adapter fully loaded.
        """

        for param in model.parameters():
            param.requires_grad = False # Freeze everything first, optional safety.

        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True # This will unfreeze only the LoRA parameters.

    @staticmethod
    def remove_peft_leftovers(model: PreTrainedModel):
        """
        If you have been getting a warning about a peft config already set, you probably didn't do:

        model.save_pretrained(path, save_peft_format=False)

        If so, this function removes the peft leftovers. And if you did save_peft_format=False and still got the error.
        Just call this function

        :param model: Your model without any Peft applied already(not a PeftModel).
        """
        for attr in ["peft_type", "active_adapters",
                     "_hf_peft_config_loaded", "peft_config"]:
            if hasattr(model, attr):
                try: delattr(model, attr)
                except AttributeError as _: pass