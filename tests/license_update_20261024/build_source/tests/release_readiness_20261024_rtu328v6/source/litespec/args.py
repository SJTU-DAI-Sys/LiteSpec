import argparse
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class SGLangBackendArgs:
    sglang_attention_backend: str = "fa3"
    sglang_mem_fraction_static: float = 0.4
    sglang_context_length: int = None
    sglang_enable_nccl_nvls: bool = False
    sglang_enable_symm_mem: bool = False
    sglang_enable_torch_compile: bool = True
    sglang_enable_dp_attention: bool = False
    sglang_enable_dp_lm_head: bool = False
    sglang_enable_piecewise_cuda_graph: bool = False
    sglang_piecewise_cuda_graph_max_tokens: int = 4096
    sglang_piecewise_cuda_graph_tokens: List[int] = None
    sglang_ep_size: int = 1
    sglang_max_running_requests: int = None  # assign based on batch size
    sglang_max_total_tokens: int = None  # assign based on batch size and seq length

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        from sglang.srt.server_args import ATTENTION_BACKEND_CHOICES

        # sglang arguments
        parser.add_argument(
            "--sglang-attention-backend",
            type=str,
            default="flashinfer",
            choices=ATTENTION_BACKEND_CHOICES,
            help="The attention backend of SGLang backend",
        )
        parser.add_argument(
            "--sglang-mem-fraction-static",
            type=float,
            default=0.4,
            help="The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors.",
        )
        parser.add_argument(
            "--sglang-context-length",
            type=int,
            default=None,
            help="The context length of the SGLang backend",
        )
        parser.add_argument(
            "--sglang-enable-nccl-nvls",
            action=argparse.BooleanOptionalAction,
            help="Enable NCCL NVLS for prefill heavy requests when available for SGLang backend",
            default=False,
        )
        parser.add_argument(
            "--sglang-enable-symm-mem",
            action=argparse.BooleanOptionalAction,
            help="Enable NCCL symmetric memory for fast collectives for SGLang backend",
            default=False,
        )
        parser.add_argument(
            "--sglang-enable-torch-compile",
            action=argparse.BooleanOptionalAction,
            help="Optimize the model with torch.compile for SGLang backend",
            default=False,
        )
        parser.add_argument(
            "--sglang-enable-dp-attention",
            action=argparse.BooleanOptionalAction,
            help="Enable DP attention for SGLang backend",
            default=False,
        )
        parser.add_argument(
            "--sglang-enable-dp-lm-head",
            action=argparse.BooleanOptionalAction,
            help="Enable piecewise CUDA graph for SGLang backend",
            default=False,
        )
        parser.add_argument(
            "--sglang-enable-piecewise-cuda-graph",
            action=argparse.BooleanOptionalAction,
            help="Enable piecewise CUDA graph for SGLang backend's prefill",
            default=False,
        )
        parser.add_argument(
            "--sglang-piecewise-cuda-graph-max-tokens",
            type=int,
            default=4096,
            help="Set the max tokens for piecewise CUDA graph for SGLang backend",
        )
        parser.add_argument(
            "--sglang-piecewise-cuda-graph-tokens",
            type=int,
            nargs="+",
            default=None,
            help="Set the list of tokens when using piecewise cuda graph for SGLang backend",
        )
        parser.add_argument(
            "--sglang-ep-size",
            type=int,
            default=1,
            help="The ep size of the SGLang backend",
        )
        parser.add_argument(
            "--sglang-max-total-tokens",
            type=int,
            default=None,
            help="The max total tokens of the SGLang backend KV pool. Defaults to batch_size * max_length.",
        )
        parser.add_argument(
            "--sglang-max-running-requests",
            type=int,
            default=None,
            help="The max running requests of the SGLang backend. Defaults to batch_size.",
        )

    @staticmethod
    def from_args(args: argparse.Namespace) -> "SGLangBackendArgs":
        return SGLangBackendArgs(
            sglang_attention_backend=args.sglang_attention_backend,
            sglang_mem_fraction_static=args.sglang_mem_fraction_static,
            sglang_context_length=args.sglang_context_length,
            sglang_enable_nccl_nvls=args.sglang_enable_nccl_nvls,
            sglang_enable_symm_mem=args.sglang_enable_symm_mem,
            sglang_enable_torch_compile=args.sglang_enable_torch_compile,
            sglang_enable_dp_attention=args.sglang_enable_dp_attention,
            sglang_enable_dp_lm_head=args.sglang_enable_dp_lm_head,
            sglang_enable_piecewise_cuda_graph=args.sglang_enable_piecewise_cuda_graph,
            sglang_piecewise_cuda_graph_max_tokens=args.sglang_piecewise_cuda_graph_max_tokens,
            sglang_piecewise_cuda_graph_tokens=args.sglang_piecewise_cuda_graph_tokens,
            sglang_ep_size=args.sglang_ep_size,
            sglang_max_running_requests=(
                args.sglang_max_running_requests
                if args.sglang_max_running_requests is not None
                else (
                    args.target_batch_size
                    if hasattr(args, "target_batch_size")
                    else None
                )
            ),
            sglang_max_total_tokens=(
                args.sglang_max_total_tokens
                if args.sglang_max_total_tokens is not None
                else (
                    args.target_batch_size * args.max_length
                    if hasattr(args, "target_batch_size")
                    and hasattr(args, "max_length")
                    else None
                )
            ),
        )

    def to_kwargs(self) -> Dict[str, Any]:
        return dict(
            attention_backend=self.sglang_attention_backend,
            mem_fraction_static=self.sglang_mem_fraction_static,
            context_length=self.sglang_context_length,
            enable_nccl_nvls=self.sglang_enable_nccl_nvls,
            enable_symm_mem=self.sglang_enable_symm_mem,
            enable_torch_compile=self.sglang_enable_torch_compile,
            enable_dp_attention=self.sglang_enable_dp_attention,
            enable_dp_lm_head=self.sglang_enable_dp_lm_head,
            enable_piecewise_cuda_graph=self.sglang_enable_piecewise_cuda_graph,
            piecewise_cuda_graph_max_tokens=self.sglang_piecewise_cuda_graph_max_tokens,
            piecewise_cuda_graph_tokens=self.sglang_piecewise_cuda_graph_tokens,
            ep_size=self.sglang_ep_size,
            max_running_requests=self.sglang_max_running_requests,
            max_total_tokens=self.sglang_max_total_tokens,
        )
