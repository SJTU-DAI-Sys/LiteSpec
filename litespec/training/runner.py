"""Complete training lifecycle: phases, logical batches, updates and saves.

Ordinary and disaggregated draft execution share ``run_draft``. Only their
input streams differ; a packet's microbatches never change accumulation counts.
"""

import os
from pathlib import Path

# Isolate each rank's Triton/Inductor/TVM compile cache so concurrent
# torchrun processes do not corrupt a shared kernel-build directory.
def configure_rank_caches():
    root = Path(__file__).resolve().parents[2]
    rank = os.environ.get("RANK", "0")
    for name, directory in (
        ("TVM_FFI_CACHE_DIR", "tvm-ffi"),
        ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
        ("TRITON_CACHE_DIR", "triton"),
    ):
        value = (
            os.environ.get(name.removesuffix("_DIR") + "_ROOT")
            or os.environ.get(name)
            or str(root / ".cache" / directory)
        )
        os.environ[name] = str(Path(value) / f"rank_{rank}")


class OptimizerWindow:
    """Sum loss terms, normalize FSDP gradients once, and aggregate telemetry."""

    def __init__(self, optimizer, accumulation_steps, process_group=None):
        from .logging import TrainingMetricAccumulator

        self.optimizer = optimizer
        self.accumulation_steps = int(accumulation_steps)
        if self.accumulation_steps <= 0:
            raise ValueError("accumulation_steps must be positive")
        self.process_group = process_group
        self.metrics = TrainingMetricAccumulator()
        self.micro_steps = 0
        self.denominator = None

    def backward(self, output):
        self.metrics.update(output)
        self.backward_terms(output.loss_terms.numerator, output.loss_terms.denominator)

    def backward_terms(self, numerator, denominator):
        (numerator / self.accumulation_steps).backward()
        value = denominator.detach()
        self.denominator = (
            value.clone() if self.denominator is None else self.denominator + value
        )
        self.micro_steps += 1

    def step(self):
        from .optimizer import normalize_accumulated_gradients

        if not self.micro_steps:
            raise RuntimeError("Cannot step an empty accumulation window")
        normalize_accumulated_gradients(
            self.optimizer,
            self.denominator,
            self.accumulation_steps,
            self.process_group,
        )
        norm = self.optimizer.step()
        self.micro_steps = 0
        self.denominator = None
        return norm


def iter_epoch(dataloader, epoch, *, start_batch=0, description=None):
    """Common deterministic epoch/cursor handling, also usable by stage runners."""
    import torch.distributed as dist
    from tqdm import tqdm

    dataloader.sampler.set_epoch(epoch)
    show_progress = not dist.is_initialized() or dist.get_rank() == 0
    iterator = tqdm(dataloader, desc=description) if show_progress else dataloader
    for index, batch in enumerate(iterator):
        if index >= start_batch:
            yield index, batch


def iter_logical_batches(session, phase):
    """Yield one complete logical batch at a time, keeping receive slots alive."""
    if session.topology is not None:
        from litespec.distributed.transport import NodePacketTransport, PacketReceiver

        transport = NodePacketTransport(
            topology=session.topology,
            routes=session.routes,
            profile=session.args.profile,
        )
        spec = session.adapter.packet_spec(session, phase)
        slots = [
            session.adapter.empty_packet(spec)
            for _ in range(session.args.pipeline_depth)
        ]
        start = session.state.global_step
        stop = (
            start
            - session.state.phase_step
            + phase.batch_count(session.steps_per_epoch)
        )
        packets = PacketReceiver(
            transport,
            slots,
            start=start,
            stop=stop,
            phase_id=phase.phase_id,
            schema_id=spec.schema_id,
        )
        for _, packet in packets:
            yield packet
        if session.args.profile:
            print(
                f"disagg draft bridge phase={phase.name} ms={packets.wait_ms:.3f}",
                flush=True,
            )
        return
    state = session.state
    for epoch in range(state.epoch, phase.epochs):
        start = state.next_batch if epoch == state.epoch else 0
        for _, data in iter_epoch(
            session.dataloader,
            epoch,
            start_batch=start,
            description=f"{phase.name} epoch {epoch}",
        ):
            if state.phase_step >= phase.batch_count(session.steps_per_epoch):
                return
            yield data


def iter_microbatch_slices(session):
    """Yield sample-axis (start, exclusive end) bounds; (None, None) uses the full batch."""
    if session.topology is None:
        yield None, None
    else:
        local_batch_size = session.args.node_batch_size // session.args.draft_ranks_per_node
        micro_batch_size = session.args.draft_micro_batch_size
        for microbatch_start in range(0, local_batch_size, micro_batch_size):
            yield microbatch_start, microbatch_start + micro_batch_size


def log_window(session, window, grad_norm):
    import torch.distributed as dist

    from .logging import build_training_log_payload

    reduced_metrics = session.adapter.log_metrics(
        session, window.metrics.reduce(session.process_group)
    )
    details = {
        "micro_step": session.state.global_step,
        "grad_norm": grad_norm,
        **session.adapter.detail_metrics(session),
    }
    prefix = session.adapter.algorithm
    if session.adapter.modes:
        prefix += "/" + session.phase.name
    payload = build_training_log_payload(
        reduced_metrics,
        session.optimizer.get_learning_rate(),
        detail_metrics=details,
        detail_prefix=prefix,
    )
    if getattr(session.optimizer, "use_muon", False):
        payload.update(
            {
                f"train/{name}_learning_rate": value
                for name, value in session.optimizer.get_learning_rates().items()
            }
        )
    session.tracker.log(payload, step=session.state.optimizer_step)
    if not dist.is_initialized() or dist.get_rank() == session.coordinator_rank:
        print(
            f"{prefix} optimizer_step={session.state.optimizer_step} "
            f"micro_step={session.state.global_step} "
            f"loss={float(reduced_metrics['objective/loss']):.4f}",
            flush=True,
        )


def run_draft(session, *, batch_source=iter_logical_batches, save=None, log=log_window):
    """The sole training loop, also exercised directly by CPU runner tests.

    ``batch_source`` yields logical batches, not forwards. All trainable work is
    performed by the recipe; this loop alone owns backward and update timing.
    """
    import torch
    import torch.distributed as dist

    from .checkpoint import restore_rng_state, save_session
    from .logging import log_cuda_peak

    save = save_session if save is None else save
    args, state = session.args, session.state
    window = OptimizerWindow(
        session.optimizer, args.accumulation_steps, session.process_group
    )
    # Restore RNG state from checkpoint
    pending_rng = session.resume_state["rng_state"] if session.resume_state is not None else None
    accumulated = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Perform a model parameter update after each gradient accumulation step.
    def update(*, tail=False):
        nonlocal accumulated
        grad_norm = window.step()
        state.optimizer_step += 1
        accumulated = 0
        if (
            tail or state.optimizer_step % args.log_interval == 0
            or (
                session.phase.log_at_schedule_end
                and state.optimizer_step == session.total_steps
            )
        ):
            log(session, window, grad_norm)
        window.metrics.reset()

    # The producer still participates in a barrier for every skipped DLite phase.
    if session.topology is not None and session.adapter.phase_barrier:
        for _ in range(state.phase_index):
            dist.barrier()
    while state.phase_index < len(session.phases):
        cur_phase = session.phase
        phase_metrics = session.adapter.enter_phase(session, cur_phase)
        if phase_metrics:
            session.tracker.log(phase_metrics, step=state.optimizer_step)
        if cur_phase.start_checkpoint and state.phase_step == 0:
            # A resumed boundary already has this checkpoint; do not duplicate it.
            if (
                session.resume_state is None
                or state.phase_index != session.resume_state["cursor"]["phase_index"]
            ):
                save(session, cur_phase.start_checkpoint)
        remaining_batches = (
            cur_phase.batch_count(session.steps_per_epoch) - state.phase_step
        )
        batch_iterator = iter(batch_source(session, cur_phase))
        for _ in range(remaining_batches):
            try:
                data = next(batch_iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    f"{cur_phase.name} input ended before the declared phase length"
                ) from exc
            # Skip/rebuild the input iterator first, then restore anchor/dropout RNG.
            if pending_rng is not None:
                restore_rng_state(pending_rng)
                pending_rng = None
            epoch = state.epoch
            collect_detailed_metrics = (
                (state.optimizer_step + 1) % args.log_interval == 0
                # Every microbatch in a final partial window is logged by
                # update(tail=True), even outside the regular interval.
                or (
                    accumulated + cur_phase.batch_count(session.steps_per_epoch)
                    - state.phase_step
                ) < args.accumulation_steps
                or (
                    cur_phase.log_at_schedule_end
                    and state.optimizer_step + 1 == session.total_steps
                )
            )
            for start, end in iter_microbatch_slices(session):
                output = session.adapter.compute_step(
                    session,
                    data,
                    microbatch_start=start,
                    microbatch_end=end,
                    collect_detailed_metrics=collect_detailed_metrics,
                )
                window.backward(output)
                del output
            state.phase_step += 1
            state.global_step += 1
            state.epoch, state.next_batch = divmod(
                state.phase_step, session.steps_per_epoch
            )
            accumulated += 1
            if accumulated == args.accumulation_steps:
                update()
            if state.global_step % args.save_interval == 0 and accumulated == 0:
                suffix = (
                    f"step_{state.global_step}"
                    if session.topology is not None
                    else f"epoch_{epoch}_step_{state.phase_step}"
                )
                save(session, cur_phase.checkpoint_prefix + suffix)
        # Resume the source once to release its last receive slot and collect profile data.
        try:
            next(batch_iterator)
        except StopIteration:
            pass
        else:
            raise RuntimeError(f"{cur_phase.name} input exceeded the declared phase length")
        if accumulated:
            update(tail=True)
        if session.topology is None:
            prefix = session.adapter.algorithm
            if session.adapter.modes:
                prefix += "/" + cur_phase.name
            memory = log_cuda_peak(cur_phase.name)
            session.tracker.log(
                {
                    f"detail/{prefix}/cuda_peak_{key}": value
                    for key, value in memory.items()
                },
                step=state.optimizer_step,
            )
        # Boundary checkpoints point to the NEXT phase, not a finished old phase.
        state.phase_index += 1
        state.epoch = state.next_batch = state.phase_step = 0
        if getattr(args, "final_save", True):
            save(session, cur_phase.final_checkpoint)
        if session.topology is not None and session.adapter.phase_barrier:
            dist.barrier()


def run_target(session):
    """Producer lifecycle; the existing transport/wire contract is unchanged."""
    import torch.distributed as dist

    from litespec.distributed.transport import NodePacketTransport

    from .setup import _broadcast_input

    args, topology = session.args, session.topology
    transport = (
        NodePacketTransport(
            topology=topology, routes=session.routes, profile=args.profile
        )
        if topology.is_target_leader
        else None
    )
    start_id = session.resume_state["cursor"]["global_step"] if session.resume_state else 0
    batch_id = 0
    for phase in session.phases:
        spec = session.adapter.packet_spec(session, phase)
        batch_id, bridge_ms = run_target_phase(
            args=args,
            topology=topology,
            dataloader=session.dataloader,
            steps=session.steps_per_epoch,
            epochs=phase.epochs,
            epoch_offset=session.adapter.target_epoch_offset(phase),
            start_id=start_id,
            batch_id=batch_id,
            limit=phase.batch_count(session.steps_per_epoch),
            transport=transport,
            phase_id=phase.phase_id,
            schema_id=spec.schema_id,
            broadcast=lambda data: _broadcast_input(args, topology, data),
            produce=lambda values, current: session.adapter.produce_packet(
                session, phase, values, current
            ),
            replay_skipped=session.adapter.replay_skipped,
        )
        if args.profile and topology.is_target_leader:
            print(
                f"disagg target bridge phase={phase.name} ms={bridge_ms:.3f}",
                flush=True,
            )
        if session.adapter.phase_barrier:
            dist.barrier()

"""Build the training session, run draft or target, then tear down process groups.
- args: parsed CLI/YAML namespace from adapter.parse_args (data, topology, hyperparameter, …).
- adapter: algorithm plugin (TrainingAdapter) for phases, models, and compute_step.
"""
def run_training(args, adapter):
    import logging
    import traceback
    import torch.distributed as dist
    from .setup import build_training_session
    from accelerate.utils import set_seed
    from litespec.distributed import destroy_distributed

    logging.basicConfig(level=logging.INFO)
    if args.disaggregate and os.environ.get("LITESPEC_DISAGG_DEBUG") == "1":
        import faulthandler
        import signal

        faulthandler.register(signal.SIGUSR1, all_threads=True)
    set_seed(args.seed)
    session = None
    completed = False
    try:
        session = build_training_session(args, adapter)
        if session.topology is not None and session.topology.is_target:
            run_target(session)
        else:
            run_draft(session)
        dist.barrier()
        completed = True
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if session is not None and session.tracker is not None:
            session.tracker.close()
        # On failure let torch elastic terminate blocked peers; collective teardown
        # could hide the original error until the distributed timeout.
        if completed:
            destroy_distributed()


def run_target_phase(
    *,
    args,
    topology,
    dataloader,
    steps,
    epochs,
    epoch_offset,
    start_id,
    batch_id,
    limit,
    transport,
    phase_id,
    schema_id,
    broadcast,
    produce,
    replay_skipped=False,
):
    """Execute target prefill and drain one packet schema before returning."""
    handles = [None] * int(args.pipeline_depth)
    progress, wait_ms = 0, 0.0
    for epoch in range(epochs):
        if topology.is_target_leader:
            dataloader.sampler.set_epoch(epoch_offset + epoch)
            iterator = iter(dataloader)
        for _ in range(steps):
            if progress >= limit:
                break
            data = next(iterator) if topology.is_target_leader else None
            slot = batch_id % len(handles)
            if batch_id >= start_id or replay_skipped:
                if topology.is_target_leader and handles[slot] is not None:
                    wait_ms += transport.wait_send(handles[slot])
                packet = produce(broadcast(data), batch_id)
                if batch_id >= start_id and topology.is_target_leader:
                    handles[slot] = transport.send(
                        packet,
                        batch_id=batch_id,
                        phase_id=phase_id,
                        schema_id=schema_id,
                    )
            batch_id += 1
            progress += 1
        if progress >= limit:
            break
    if topology.is_target_leader:
        for handle in handles:
            if handle is not None:
                wait_ms += transport.wait_send(handle)
    return batch_id, wait_ms
