"""Rank-zero instrumentation loaded only by this trial's vLLM worker extension.

Counts actual top-level model forward calls, not HTTP requests or TP ranks.
Token IDs, images and hidden states are never written to the trace.
"""
import functools
import json
import os
import time

from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class TraceWorkerExtension:
    pass


_load_model = GPUModelRunner.load_model
_execute_model = GPUModelRunner.execute_model


@functools.wraps(_load_model)
def load_model(self, *args, **kwargs):
    result = _load_model(self, *args, **kwargs)
    self._jev_current_step = None

    def count_model(module, args):
        if self._jev_current_step is not None:
            self._jev_current_step["model_forward_calls"] += 1

    def count_vision(module, args):
        if self._jev_current_step is not None:
            self._jev_current_step["vision_forward_calls"] += 1

    self.model.register_forward_pre_hook(count_model)
    if getattr(self.model, "visual", None) is not None:
        self.model.visual.register_forward_pre_hook(count_vision)
    return result


@functools.wraps(_execute_model)
def execute_model(self, scheduler_output, *args, **kwargs):
    import torch.distributed as dist

    if dist.get_rank() != 0:
        return _execute_model(self, scheduler_output, *args, **kwargs)
    new = {r.req_id: r for r in scheduler_output.scheduled_new_reqs}
    cached = scheduler_output.scheduled_cached_reqs
    computed = dict(zip(cached.req_ids, cached.num_computed_tokens))
    requests = []
    for req_id, count in scheduler_output.num_scheduled_tokens.items():
        if req_id in new:
            state = new[req_id]
            prompt_length = len(state.prompt_token_ids)
            previous = state.num_computed_tokens
        else:
            state = self.requests[req_id]
            prompt_length = len(state.prompt_token_ids)
            previous = computed.get(req_id, state.num_computed_tokens)
        prefill = min(count, max(0, prompt_length - previous))
        requests.append({"id": req_id, "prompt_tokens": prompt_length,
                         "computed_before": previous, "scheduled_tokens": count,
                         "prefill_tokens": prefill, "decode_tokens": count - prefill})
    record = {"time": time.time(), "rank": 0, "requests": requests,
              "model_forward_calls": 0, "vision_forward_calls": 0}
    self._jev_current_step = record
    try:
        return _execute_model(self, scheduler_output, *args, **kwargs)
    finally:
        self._jev_current_step = None
        record["prefill_tokens"] = sum(r["prefill_tokens"] for r in requests)
        record["decode_tokens"] = sum(r["decode_tokens"] for r in requests)
        path = os.environ.get("JEV_FORWARD_TRACE")
        if path and requests:
            with open(path, "a") as handle:
                handle.write(json.dumps(record) + "\n")


GPUModelRunner.load_model = load_model
GPUModelRunner.execute_model = execute_model
