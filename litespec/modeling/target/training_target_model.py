from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoModelForCausalLM

from litespec.distributed import get_tp_group


@dataclass
class TargetTrainingOutput:
    hidden_states: Union[
        torch.Tensor, Tuple[torch.Tensor, ...], Dict[int, torch.Tensor]
    ]
    logits: Optional[torch.Tensor]  # [batch, seq_len, vocab], optional in Stage 1
    input_ids: torch.Tensor  # [batch, seq_len]
    attention_mask: torch.Tensor  # [batch, seq_len]
    loss_mask: torch.Tensor  # [batch, seq_len]


class TrainingTargetModel(ABC):
    """
    Abstract backend for extracting target-model training features.
    """

    def __init__(self):
        self.capture_layer_ids = None

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "TrainingTargetModel":
        """Initialize the target model backend."""

    @abstractmethod
    def generate_training_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        return_logits: bool = True,
    ) -> TargetTrainingOutput:
        """Generate target hidden states and optional logits for draft training."""

    def set_capture_layers(self, layer_ids: List[int]) -> None:
        """Set which layers' hidden states to capture."""
        self.capture_layer_ids = layer_ids


class SGLangTrainingTargetModel(TrainingTargetModel):
    def __init__(self, model_runner):
        super().__init__()
        self.model_runner = model_runner

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "SGLangTrainingTargetModel":
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.server_args import ServerArgs

        from .sglang_backend import (
            SGLangRunner,
            wrap_eagle3_logits_processors_in_module,
        )

        tp_size = dist.get_world_size(get_tp_group())
        dtype_arg = torch_dtype if torch_dtype is not None else "auto"
        server_args = ServerArgs(
            model_path=pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            dtype=dtype_arg,
            enable_return_hidden_states=True,
            disable_cuda_graph=True,
            tp_size=tp_size,
            pp_size=1,
            **kwargs,
        )

        tp_rank = dist.get_rank(get_tp_group())
        moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)
        model_config = ModelConfig.from_server_args(server_args)

        model_runner = SGLangRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=torch.cuda.current_device(),
            tp_rank=dist.get_rank(get_tp_group()),
            tp_size=server_args.tp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=server_args.ep_size,
            pp_rank=0,
            pp_size=1,
            server_args=server_args,
            nccl_port=None,
            is_draft_worker=False,
        )
        wrap_eagle3_logits_processors_in_module(
            model_runner.model, return_full_logits=True
        )
        instance = cls(model_runner)
        # Default to all layers until the caller selects the required subset.
        instance.set_capture_layers([])
        return instance

    def set_capture_layers(self, layer_ids: List[int]) -> None:
        """Set which layers' hidden states to capture.

        If layer_ids is None or empty, capture all layers. SGLang's capture
        APIs add a +1 offset to HF-style layer indices; the final layer is
        returned separately as ``last_hidden_states``.
        """
        num_layers = getattr(self.model_runner.model_config, "num_hidden_layers", 36)
        if layer_ids is None or len(layer_ids) == 0:
            layer_ids = list(range(num_layers))

        super().set_capture_layers(layer_ids)
        model = self.model_runner.model
        if hasattr(model, "set_eagle3_layers_to_capture"):
            model.set_eagle3_layers_to_capture(layer_ids)
            return
        if hasattr(model, "set_dflash_layers_to_capture"):
            # Qwen3.5 exposes this hook in newer SGLang versions. The final
            # normalized layer is already supplied by the logits processor.
            model.set_dflash_layers_to_capture(
                [layer_id for layer_id in layer_ids if layer_id < num_layers - 1]
            )
            return
        self._set_communicator_capture(model, layer_ids, num_layers)

    @staticmethod
    def _set_communicator_capture(
        model: nn.Module, layer_ids: List[int], num_layers: int
    ) -> None:
        """Install hidden-state capture for stock SGLang 0.5.9 Qwen3.5."""
        language_model = getattr(model, "model", None)
        layers = getattr(language_model, "layers", None)
        if layers is None or len(layers) != num_layers:
            raise RuntimeError(
                f"SGLang model {type(model).__name__} does not expose a supported "
                "hidden-state capture hook or compatible decoder layers."
            )

        state = getattr(model, "_training_target_communicator_capture", None)
        if state is None:
            state = {"selected_next_layer_ids": set(), "hidden_states": []}
            model._training_target_communicator_capture = state

            for next_layer_id, layer in enumerate(layers):
                communicator = getattr(layer, "layer_communicator", None)
                original_prepare_attn = getattr(communicator, "prepare_attn", None)
                communicate_simple = getattr(
                    communicator, "_communicate_simple_fn", None
                )
                if original_prepare_attn is None or communicate_simple is None:
                    raise RuntimeError(
                        "SGLang Qwen3.5 decoder layer does not expose the expected "
                        "LayerCommunicator interface."
                    )

                def prepare_attn_and_capture(
                    hidden_states,
                    residual,
                    forward_batch,
                    *args,
                    _next_layer_id=next_layer_id,
                    _communicator=communicator,
                    _original_prepare_attn=original_prepare_attn,
                    **kwargs,
                ):
                    hidden_states, residual = _original_prepare_attn(
                        hidden_states,
                        residual,
                        forward_batch,
                        *args,
                        **kwargs,
                    )
                    if _next_layer_id in state["selected_next_layer_ids"]:
                        captured = _communicator._communicate_simple_fn(
                            hidden_states=residual,
                            forward_batch=forward_batch,
                            context=_communicator._context,
                        )
                        if captured is residual:
                            captured = residual.clone()
                        state["hidden_states"].append(captured)
                    return hidden_states, residual

                communicator.prepare_attn = prepare_attn_and_capture

            from .sglang_backend.utils import LogitsProcessorForEAGLE3

            processors = [
                module
                for module in model.modules()
                if isinstance(module, LogitsProcessorForEAGLE3)
            ]
            if not processors:
                raise RuntimeError(
                    "Could not attach Qwen3.5 hidden-state capture to the SGLang "
                    "logits processor."
                )

            def take_hidden_states(state=state):
                hidden_states = state["hidden_states"]
                state["hidden_states"] = []
                return hidden_states

            for processor in processors:
                processor.aux_hidden_states_provider = take_hidden_states

        state["selected_next_layer_ids"] = {
            layer_id + 1 for layer_id in layer_ids if layer_id < num_layers - 1
        }

    @staticmethod
    def _unpack_runner_output(runner_output):
        if isinstance(runner_output, tuple):
            return runner_output[0]
        if (
            hasattr(runner_output, "logits_output")
            and runner_output.logits_output is not None
        ):
            return runner_output.logits_output
        return runner_output

    @staticmethod
    def _split_aux_and_last_capture_ids(
        captured_ids: List[int],
        num_aux_layers: int,
        num_transformer_layers: Optional[int],
        has_last_hidden: bool,
    ) -> Tuple[List[int], List[int]]:
        """Map layer ids to SGLang aux vs last_hidden capture buckets.

        SGLang's EAGLE3 hook records layer ``L`` in ``aux_hidden_states`` only when
        ``L < num_transformer_layers - 1``. The final transformer layer is returned
        via ``last_hidden_states`` instead.
        """
        if num_transformer_layers is None:
            if captured_ids:
                if has_last_hidden and len(captured_ids) == num_aux_layers + 1:
                    return captured_ids[:-1], captured_ids[-1:]
                return captured_ids, []
            return list(range(num_aux_layers)), []

        final_layer_id = num_transformer_layers - 1
        if captured_ids:
            aux_capture_ids = [
                layer_id for layer_id in captured_ids if layer_id < final_layer_id
            ]
            last_capture_ids = [
                layer_id for layer_id in captured_ids if layer_id == final_layer_id
            ]
        else:
            aux_capture_ids = list(range(final_layer_id))
            last_capture_ids = [final_layer_id] if has_last_hidden else []

        if len(aux_capture_ids) != num_aux_layers:
            raise ValueError(
                f"Expected {len(aux_capture_ids)} aux hidden layers from SGLang, "
                f"got {num_aux_layers} (captured_ids={captured_ids})."
            )
        if last_capture_ids and not has_last_hidden:
            raise ValueError(
                "Missing last_hidden_states for final captured layer(s) "
                f"{last_capture_ids}."
            )
        return aux_capture_ids, last_capture_ids

    def _aux_hidden_to_layer_tuple(
        self,
        aux_hidden: torch.Tensor,
        last_hidden: Optional[torch.Tensor] = None,
        num_layers: Optional[int] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Split SGLang concatenated aux hidden states into per-layer tensors."""
        hidden_size = self.model_runner.model_config.hidden_size
        bsz, seq_len, total_h = aux_hidden.shape
        if num_layers is None:
            if total_h % hidden_size != 0:
                raise ValueError(
                    f"Unexpected aux_hidden_states shape {tuple(aux_hidden.shape)}; "
                    f"last dim must be divisible by hidden_size={hidden_size}"
                )
            num_layers = total_h // hidden_size
        elif total_h != int(num_layers) * hidden_size:
            raise ValueError(
                f"aux_hidden_states shape {tuple(aux_hidden.shape)} does not match "
                f"{num_layers} captured layers at hidden_size={hidden_size}."
            )
        reshaped = aux_hidden.view(bsz, seq_len, num_layers, hidden_size)
        layers = [reshaped[:, :, layer_idx, :] for layer_idx in range(num_layers)]
        if last_hidden is not None:
            if last_hidden.shape != (bsz, seq_len, hidden_size):
                raise ValueError(
                    f"Unexpected last_hidden_states shape {tuple(last_hidden.shape)}; "
                    f"expected {(bsz, seq_len, hidden_size)}"
                )
            layers.append(last_hidden)
        return tuple(layers)

    @torch.no_grad
    def _extend(self, reqs, *, return_logits: bool):
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.radix_cache import RadixCache
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
        )
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang.srt.utils import require_mlp_sync, require_mlp_tp_gather

        from .sglang_backend.utils import LogitsProcessorForEAGLE3

        for _, module in self.model_runner.model.named_modules():
            if isinstance(module, LogitsProcessorForEAGLE3):
                module.return_last_hidden_states = True
                module.return_logits = return_logits

        cache_params = CacheInitParams(
            disable=False,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            page_size=self.model_runner.server_args.page_size,
        )
        tree_cache = RadixCache(cache_params)

        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=self.model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()

        if require_mlp_sync(self.model_runner.server_args):
            Scheduler.prepare_mlp_sync_batch_raw(
                batch,
                dp_size=self.model_runner.server_args.dp_size,
                attn_tp_size=1,
                tp_group=self.model_runner.tp_group,
                get_idle_batch=None,
                disable_cuda_graph=self.model_runner.server_args.disable_cuda_graph,
                spec_algorithm=SpeculativeAlgorithm.NONE,
                speculative_num_draft_tokens=None,
                require_mlp_tp_gather=require_mlp_tp_gather(
                    self.model_runner.server_args
                ),
                disable_overlap_schedule=self.model_runner.server_args.disable_overlap_schedule,
                offload_tags=set(),
            )

        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL

        runner_output = self.model_runner.forward(forward_batch)
        output = self._unpack_runner_output(runner_output)

        input_lens = [len(req.origin_input_ids) for req in reqs]
        last_hidden_list = None
        if (
            hasattr(output, "aux_hidden_states")
            and output.aux_hidden_states is not None
        ):
            hidden_states_list = torch.split(
                output.aux_hidden_states, input_lens, dim=0
            )
            if (
                hasattr(output, "last_hidden_states")
                and output.last_hidden_states is not None
            ):
                last_hidden_list = torch.split(
                    output.last_hidden_states, input_lens, dim=0
                )
        elif hasattr(output, "hidden_states") and output.hidden_states is not None:
            hidden_states_list = torch.split(output.hidden_states, input_lens, dim=0)
        else:
            raise ValueError("SGLang output does not contain hidden states.")

        logits_list = None
        if return_logits:
            if not hasattr(output, "logits") or output.logits is None:
                raise ValueError("SGLang output does not contain full prefill logits.")
            logits_list = torch.split(output.logits, input_lens, dim=0)

        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool_allocator.clear()

        return hidden_states_list, last_hidden_list, logits_list

    @torch.no_grad()
    def generate_training_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        return_logits: bool = True,
    ) -> TargetTrainingOutput:
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        sampling_params = SamplingParams(temperature=0, max_new_tokens=1)
        reqs, data_cache = [], []

        if isinstance(input_ids, torch.Tensor):
            input_ids_list = torch.split(input_ids, 1, dim=0)
            attn_mask_list = torch.split(attention_mask, 1, dim=0)
            loss_mask_list = torch.split(loss_mask, 1, dim=0)

        for idx, (curr_ids, curr_attn, curr_loss) in enumerate(
            zip(input_ids_list, attn_mask_list, loss_mask_list)
        ):
            req = Req(
                rid=str(idx),
                origin_input_text="",
                origin_input_ids=curr_ids.view(-1).tolist(),
                sampling_params=sampling_params,
            )
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
            data_cache.append((curr_ids, curr_attn, curr_loss))
            reqs.append(req)

        hidden_states_list, last_hidden_list, logits_list = self._extend(
            reqs, return_logits=return_logits
        )

        batched_aux = torch.cat([h.unsqueeze(0) for h in hidden_states_list], dim=0)
        batched_last = None
        if last_hidden_list is not None:
            batched_last = torch.cat([h.unsqueeze(0) for h in last_hidden_list], dim=0)

        hidden_size = self.model_runner.model_config.hidden_size
        num_transformer_layers = getattr(
            self.model_runner.model_config, "num_hidden_layers", None
        )
        captured_ids = list(self.capture_layer_ids or [])
        expected_aux_count = None
        if num_transformer_layers is not None:
            final_layer_id = num_transformer_layers - 1
            expected_aux_count = sum(
                layer_id < final_layer_id for layer_id in captured_ids
            )
        if expected_aux_count == 0 and batched_last is not None:
            # When only the final layer is requested, SGLang mirrors its final
            # state into aux_hidden_states as a fallback. It is not a second
            # captured layer and must not be serialized twice.
            num_aux_layers = 0
            aux_layer_tensors = ()
        elif (
            batched_aux.ndim == 3
            and batched_aux.shape[-1] % hidden_size == 0
            and batched_aux.shape[-1] >= hidden_size
        ):
            num_aux_layers = batched_aux.shape[-1] // hidden_size
            aux_layer_tensors = self._aux_hidden_to_layer_tuple(
                batched_aux,
                last_hidden=None,
                num_layers=num_aux_layers,
            )
        else:
            raise ValueError(
                "SGLang returned invalid aux hidden states; expected one or more "
                "captured layers aligned to hidden_size."
            )
        aux_capture_ids, last_capture_ids = self._split_aux_and_last_capture_ids(
            captured_ids,
            num_aux_layers,
            num_transformer_layers,
            batched_last is not None,
        )
        if num_transformer_layers is not None and (
            not captured_ids or len(captured_ids) < num_transformer_layers
        ):
            # Partial capture: map absolute layer id -> hidden tensor.
            hidden_states = {
                layer_id: aux_layer_tensors[idx]
                for idx, layer_id in enumerate(aux_capture_ids)
            }
            for layer_id in last_capture_ids:
                hidden_states[layer_id] = batched_last
        else:
            layer_tensors = list(aux_layer_tensors)
            if batched_last is not None:
                layer_tensors.append(batched_last)
            hidden_states = tuple(layer_tensors)
        input_ids = torch.cat([d[0] for d in data_cache], dim=0)
        attention_mask = torch.cat([d[1] for d in data_cache], dim=0)
        loss_mask = torch.cat([d[2] for d in data_cache], dim=0)
        logits = (
            torch.cat([value.unsqueeze(0) for value in logits_list], dim=0)
            if logits_list is not None
            else None
        )

        return TargetTrainingOutput(
            hidden_states=hidden_states,
            logits=logits,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )


class HFTrainingTargetModel(TrainingTargetModel):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "HFTrainingTargetModel":

        target_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            output_hidden_states=True,
            trust_remote_code=trust_remote_code,
            **kwargs,
        ).eval()

        if device:
            target_model = target_model.to(device)

        return cls(target_model)

    @torch.no_grad()
    def generate_training_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        return_logits: bool = True,
    ) -> TargetTrainingOutput:
        forward_model = self.model
        if not return_logits:
            base_model = getattr(self.model, "model", None)
            if isinstance(base_model, nn.Module) and base_model is not self.model:
                forward_model = base_model
        outputs = forward_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # hidden_states[0] = embedding output; hidden_states[i+1] = layer i output
        # Take all layers except embedding (index 0) and concat in feature dim
        all_hidden_states = outputs.hidden_states[1:]
        capture_ids = list(self.capture_layer_ids or [])
        hidden_states = (
            {layer_id: all_hidden_states[layer_id] for layer_id in capture_ids}
            if capture_ids
            else all_hidden_states
        )

        return TargetTrainingOutput(
            hidden_states=hidden_states,
            logits=outputs.logits if return_logits else None,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )


def get_training_target_model(
    pretrained_model_name_or_path: str,
    backend: str = "sglang",
    torch_dtype: torch.dtype = None,
    device: str = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> TrainingTargetModel:
    if backend == "sglang":
        # Import sglang only when the sglang backend is requested.
        return SGLangTrainingTargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    elif backend == "hf":
        return HFTrainingTargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")
