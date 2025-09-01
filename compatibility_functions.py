import warnings
import numpy as np
import torch
import datasets
from datasets import IterableDataset
from typing import Any, Literal, Optional, Union
from transformers import DataCollatorForLanguageModeling

class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only calculated on the completion made by
    the assistant.

    Args:
        response_template (`Union[str, list[int]]`):
            the template form that indicates the start of the response, typically something like '### Response:\n'. It
            can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        instruction_template (`Union[str, list[int]]`):
            the template form that indicates the start of the human instruction, typically something like '###
            Human:\n'. Useful for assistant-style conversation datasets. It can also be passed as tokenized ids.
        mlm (`bool`, *optional*, defaults to `False`): Whether to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        response_template: Union[str, list[int]],
        instruction_template: Optional[Union[str, list[int]]] = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        padding_free: bool = False,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)
        warnings.warn(
            "This class is deprecated and will be removed in version 0.20.0. To train on completion only, please use "
            "the parameter `completion_only_loss` of `SFTConfig` instead.",
            DeprecationWarning,
        )

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template

        self.response_template = response_template
        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template

        if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value.",
                UserWarning,
            )

        self.ignore_index = ignore_index
        self.padding_free = padding_free

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        batch = super().torch_call(examples)

        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None

                for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                    if (
                        self.response_token_ids
                        == batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the following instance: "
                        f"{self.tokenizer.decode(batch['input_ids'][i])}. This instance will be ignored in loss "
                        "calculation. Note, if this happens often, consider increasing the `max_length`.",
                        UserWarning,
                    )
                    batch["labels"][i, :] = self.ignore_index
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

                    # Make pytorch loss function ignore all tokens up through the end of the response key
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        else:
            for i in range(len(examples)):
                response_token_ids_idxs = []
                human_token_ids_idxs = []

                for assistant_idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # find the indexes of the start of a response.
                    if (
                        self.response_token_ids
                        == batch["labels"][i][assistant_idx : assistant_idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_idxs.append(assistant_idx + len(self.response_token_ids))

                if len(response_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the following instance: "
                        f"{self.tokenizer.decode(batch['input_ids'][i])}. This instance will be ignored in loss "
                        "calculation. Note, if this happens often, consider increasing the `max_length`.",
                        UserWarning,
                    )
                    batch["labels"][i, :] = self.ignore_index

                human_token_ids = self.instruction_token_ids
                for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
                    # find the indexes of the start of a human answer.
                    if human_token_ids == batch["labels"][i][human_idx : human_idx + len(human_token_ids)].tolist():
                        human_token_ids_idxs.append(human_idx)

                if len(human_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find instruction key `{self.instruction_template}` in the following instance: "
                        f"{self.tokenizer.decode(batch['input_ids'][i])}. This instance will be ignored in loss "
                        "calculation. Note, if this happens often, consider increasing the `max_length`.",
                        UserWarning,
                    )
                    batch["labels"][i, :] = self.ignore_index

                if (
                    len(human_token_ids_idxs) > 0
                    and len(response_token_ids_idxs) > 0
                    and human_token_ids_idxs[0] > response_token_ids_idxs[0]
                ):
                    human_token_ids_idxs = [0] + human_token_ids_idxs

                for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
                    # Make pytorch loss function ignore all non response tokens
                    if idx != 0:
                        batch["labels"][i, start:end] = self.ignore_index
                    else:
                        batch["labels"][i, :end] = self.ignore_index

                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index

        if self.padding_free:
            # remove padding, `attention_mask` and add `position_ids`
            attn_mask = batch.pop("attention_mask")
            batch["input_ids"] = batch["input_ids"][attn_mask.bool()].unsqueeze(0)
            batch["position_ids"] = attn_mask.cumsum(1)[attn_mask.bool()].unsqueeze(0) - 1
            batch["labels"] = batch["labels"][attn_mask.bool()].unsqueeze(0)
            batch["labels"][batch["position_ids"] == 0] = self.ignore_index

            # Calculate cumulative sequence lengths for queries and keys to prevent graph breaks during further computations.
            flattened_position_ids = batch["position_ids"].flatten()
            indices_q = torch.arange(
                flattened_position_ids.size(0), device=flattened_position_ids.device, dtype=torch.int32
            )
            batch["cu_seq_lens_q"] = torch.cat(
                (
                    indices_q[flattened_position_ids == 0],
                    torch.tensor(
                        flattened_position_ids.size(), device=flattened_position_ids.device, dtype=torch.int32
                    ),
                )
            ).unsqueeze(0)
            batch["cu_seq_lens_k"] = batch["cu_seq_lens_q"]

            # Determine maximum sequence lengths to prevent graph breaks during further computations.
            batch["max_length_k"] = torch.tensor([flattened_position_ids.max().item() + 1])
            batch["max_length_q"] = batch["max_length_k"]

        return batch

class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files. The dataset also formats
    the text before tokenization with a specific format that is provided by the user.

    Args:
        tokenizer (`transformers.PreTrainedTokenizer`):
            The processor used for processing the data.
        dataset (`dataset.Dataset`):
            Dataset with text files.
        dataset_text_field (`str` or `None`, *optional*, defaults to `None`):
            Name of the field in the dataset that contains the text. Only one of `dataset_text_field` and
            `formatting_func` should be provided.
        formatting_func (`Callable`, *optional*):
            Function that formats the text before tokenization. Usually it is recommended to follow a certain pattern
            such as `"### Question: {question} ### Answer: {answer}"`. Only one of `dataset_text_field` and
            `formatting_func` should be provided.
        infinite (`bool`, *optional*, defaults to `False`):
            If True the iterator is reset after dataset reaches end else stops.
        seq_length (`int`, *optional*, defaults to `1024`):
            Length of token sequences to return.
        num_of_sequences (`int`, *optional*, defaults to `1024`):
            Number of token sequences to keep in buffer.
        chars_per_token (`int`, *optional*, defaults to `3.6`):
            Number of characters per token used to estimate number of tokens in text buffer.
        eos_token_id (`int`, *optional*, defaults to `0`):
            Id of the end of sequence token if the passed tokenizer does not have an EOS token.
        shuffle (`bool`, *optional*, defaults to `True`)
            Shuffle the examples before they are returned
        append_concat_token (`bool`, *optional*, defaults to `True`)
            If true, appends `eos_token_id` at the end of each sample being packed.
        add_special_tokens (`bool`, *optional*, defaults to `True`)
            If true, tokenizers adds special tokens to each sample being packed.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        dataset_text_field=None,
        formatting_func=None,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        eos_token_id=0,
        shuffle=True,
        append_concat_token=True,
        add_special_tokens=True,
    ):
        warnings.warn(
            "This class is deprecated and will be removed in version 0.20.0. To use packing, use the argument "
            "`packing` of `SFTConfig` instead.",
            DeprecationWarning,
        )
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.shuffle = shuffle
        self.append_concat_token = append_concat_token
        self.add_special_tokens = add_special_tokens

        if dataset_text_field is not None and formatting_func is not None:
            warnings.warn(
                "Only one of `dataset_text_field` and `formatting_func` should be provided. "
                "Ignoring `dataset_text_field` and using `formatting_func`.",
                UserWarning,
            )

        if formatting_func is not None:
            self.formatting_func = formatting_func
        elif dataset_text_field is not None:
            self.formatting_func = lambda x: x[dataset_text_field]
        else:  # neither is provided
            raise ValueError("Either `dataset_text_field` or `formatting_func` should be provided.")

        self.pretokenized = False
        column_names = (
            dataset.column_names if isinstance(dataset, (datasets.Dataset, datasets.IterableDataset)) else None
        )
        if column_names is not None and "input_ids" in column_names:
            self.pretokenized = True
            # since the dataset is tokenized, the unit of buffer size should be tokens
            self.max_buffer_size = seq_length * num_of_sequences

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(self.formatting_func(next(iterator)))
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            if self.shuffle:
                random.shuffle(buffer)
            if self.pretokenized:
                tokenized_inputs = buffer
            else:
                tokenized_inputs = self.tokenizer(
                    buffer, add_special_tokens=self.add_special_tokens, truncation=False
                )["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                if self.append_concat_token:
                    tokenized_input = tokenized_input + [self.concat_token_id]
                all_token_ids.extend(tokenized_input)
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            if self.shuffle:
                # Shuffle again, otherwise split examples occur in consecutive tensors.
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }