"""LiteSpec training framework."""


def run_training(algorithm, *, mode=None):
    from .runner import configure_rank_caches

    configure_rank_caches()
    from .algorithms import get_training_adapter

    adapter = get_training_adapter(algorithm)
    args = adapter.parse_args(mode)
    from torch.distributed.elastic.multiprocessing.errors import record

    from .runner import run_training as run

    record(run)(args, adapter)
