"""RGB order ablation; invokes VJev.select unchanged for every measured image."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vjev import VJev
from games import ACTIONS, Game, SPECS, cases

ORDERS = ("system_image_question", "system_question_image", "image_rules_question")
CONTROLLERS = ("choice_logits", "binary_yes_no", "native")


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--game", choices=SPECS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--verify-detokenize", action="store_true")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    for name in ("images", "calls"):
        (out / name).mkdir()
    os.environ["JEV_FORWARD_TRACE"] = str(out / "forward-trace.jsonl")
    import torch
    import transformers
    import vllm

    model = VJev(
        args.model, dtype="bfloat16", tensor_parallel_size=1,
        max_model_len=4096, max_num_seqs=8, max_num_batched_tokens=4096,
        gpu_memory_utilization=0.5, enforce_eager=True,
        enable_prefix_caching=True, mamba_cache_mode="align", mamba_ssm_cache_dtype="float32",
        limit_mm_per_prompt={"image": 1, "video": 0},
        worker_extension_cls="jev_trace.TraceWorkerExtension",
    )
    tokenizer = model.tokenizer
    labels = list(ACTIONS)
    label_ids = {model._token_id(label): label for label in labels}
    image_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    manifest = [c for c in cases(args.n) if c["game"] == args.game]
    dump(out / "settings.json", {
        "model": args.model, "adapter": None, "dtype": "BF16", "ssm_dtype": "float32", "thinking": False,
        "vllm": vllm.__version__, "torch": torch.__version__,
        "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "orders": ORDERS, "controllers": CONTROLLERS, "manifest": manifest,
        "task": SPECS[args.game]["task"], "choices": ACTIONS,
        "cache": "Engine KV/MM/encoder caches cleared per unique RGB/order; identical calls memoized for accuracy only.",
        "batch": "One native MCQ plus four candidate inputs per select(binary_yes_no). Choice logits read from the same native MCQ forward.",
        "role_confound": "image_rules_question moves identical rule text from system into user; other two orders preserve roles.",
        "input": "Only RGB, fixed task/rules, and choices. No state/history/oracle pruning.",
        "text": "Full vocabulary logits returned without decoding vocabulary strings; generated token IDs decoded locally without repair.",
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), Path(__file__).with_name("games.py"),
                                    Path(__file__).parents[1] / "__init__.py")},
    })
    (out / "chat-template.jinja").write_text(tokenizer.chat_template)
    for p in (Path(__file__), Path(__file__).with_name("games.py")):
        (out / p.name).write_bytes(p.read_bytes())
    (out / "vjev.py").write_bytes((Path(__file__).parents[1] / "__init__.py").read_bytes())

    if args.verify_detokenize:
        original_chat = model.llm.chat
        engine_texts = []
        def capture_chat(*a, **kw):
            outputs = original_chat(*a, **kw)
            engine_texts[:] = [o.outputs[0].text for o in outputs]
            return outputs
        model.llm.chat = capture_chat
        comparisons = []
        for order in ORDERS:
            results = []
            for enabled in (True, False):
                model.params.detokenize = enabled
                model.llm.reset_prefix_cache()
                model.llm.reset_mm_cache()
                model.llm.llm_engine.engine_core.reset_encoder_cache()
                rgb = Game(args.game, manifest[0]["seed"]).render()
                result = model.select("data:image/png;base64," + base64.b64encode(rgb).decode(),
                                      SPECS[args.game]["task"], ACTIONS,
                                      mode="binary_yes_no", order=order)
                if enabled:
                    assert engine_texts == [r["text"] for r in [result["native"], *result["candidate_outputs"]]]
                results.append(result)
            assert results[0] == results[1], "Text decoding toggle changed model results"
            comparisons.append({"order": order, "identical": True, "result": results[1]})
        model.llm.chat = original_chat
        dump(out / "detokenize-equivalence.json", comparisons)

    def evaluate(rgb, order):
        digest = hashlib.sha256(rgb).hexdigest()
        call_id = f"{order}-{digest}"
        path = out / "calls" / f"{call_id}.json"
        if path.exists():
            return json.loads(path.read_text())
        (out / "images" / f"{digest}.png").write_bytes(rgb)
        assert model.llm.reset_prefix_cache() is True
        model.llm.reset_mm_cache()
        model.llm.llm_engine.engine_core.reset_encoder_cache()
        trace = out / "forward-trace.jsonl"
        offset = trace.stat().st_size if trace.exists() else 0
        started = time.perf_counter()
        result = model.select(
            "data:image/png;base64," + base64.b64encode(rgb).decode(),
            SPECS[args.game]["task"], ACTIONS, mode="binary_yes_no", order=order,
        )
        seconds = time.perf_counter() - started
        with trace.open("rb") as f:
            f.seek(offset)
            traces = [json.loads(line) for line in f.read().splitlines()]
        assert traces and sum(r["decode_tokens"] for r in traces) == 0
        audited = []
        for messages, raw in zip(result["requests"], [result["native"], *result["candidate_outputs"]]):
            expected = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_dict=False,
            )
            actual = raw["prompt_token_ids"]
            assert expected.count(image_id) == 1
            pos = expected.index(image_id)
            image_tokens = actual.count(image_id)
            assert image_tokens > 0
            assert actual == expected[:pos] + [image_id] * image_tokens + expected[pos + 1:]
            rendered = tokenizer.decode(actual)
            rule_pos = rendered.index("Task:")
            vision_pos = rendered.index("<|vision_start|>")
            question_pos = rendered.index("Choose the correct option." if not audited else "Candidate:")
            positions = (rule_pos, vision_pos, question_pos)
            if order == ORDERS[0]: assert rule_pos < vision_pos < question_pos
            elif order == ORDERS[1]: assert rule_pos < question_pos < vision_pos
            else: assert vision_pos < rule_pos < question_pos
            assert len(raw["token_ids"]) == 1
            audited.append({"rule_image_question_positions": positions,
                            "prompt_tokens": len(actual), "image_tokens": image_tokens})
        single = labels[max(range(len(labels)), key=result["native"]["logits"].__getitem__)]
        native = label_ids.get(result["native"]["token_ids"][0])
        record = {"id": call_id, "image_sha256": digest, "order": order,
                  "predictions": {"choice_logits": single, "binary_yes_no": result["choice"], "native": native},
                  "audit": audited, "seconds": seconds, "trace": traces, "result": result}
        dump(path, record)
        return record

    episodes = []
    initial = []
    for case in manifest:
        first = Game(args.game, case["seed"])
        for order in ORDERS:
            rec = evaluate(first.render(), order)
            initial.append({"case": case["index"], "order": order,
                            "optimal": first.optimal(), "call": rec["id"],
                            "predictions": rec["predictions"]})
        pairs = [(order, controller) for order in ORDERS for controller in CONTROLLERS]
        shift = case["index"] % len(pairs)
        for order, controller in pairs[shift:] + pairs[:shift]:
            game = Game(args.game, case["seed"])
            steps = []
            blocked = 0
            reason = "step_limit"
            for step in range(SPECS[args.game]["steps"]):
                rec = evaluate(game.render(), order)
                action = rec["predictions"][controller]
                row = {"step": step, "call": rec["id"], "action": action,
                       "state": game.state(), "optimal": game.optimal()}
                steps.append(row)
                if action is None:
                    reason = "invalid_native_token"
                    break
                row["execution"] = game.step(action)
                blocked = blocked + 1 if row["execution"]["blocked"] else 0
                if game.done:
                    reason = "success" if game.success else "wrong_answer"
                    break
                if blocked >= 3:
                    reason = "stalled"
                    break
            episode = {"case": case, "order": order, "controller": controller,
                       "success": game.success, "reason": reason, "steps": steps}
            episodes.append(episode)
            with (out / "episodes.jsonl").open("a") as f:
                f.write(json.dumps(episode) + "\n")
        dump(out / "initial.json", initial)
        dump(out / "progress.json", {"completed_cases": case["index"] + 1,
                                     "episodes": len(episodes), "time": time.time()})
        print(args.game, case["index"] + 1, "cases completed", flush=True)
    metrics = {}
    for order in ORDERS:
        metrics[order] = {}
        for controller in CONTROLLERS:
            group = [e for e in episodes if e["order"] == order and e["controller"] == controller]
            firsts = [r for r in initial if r["order"] == order]
            metrics[order][controller] = {
                "success": sum(e["success"] for e in group), "episodes": len(group),
                "initial_correct": sum(r["predictions"][controller] in r["optimal"] for r in firsts),
                "invalid_episodes": sum(e["reason"] == "invalid_native_token" for e in group),
            }
    dump(out / "metrics.json", metrics)
    dump(out / "completion.json", {"complete": True, "episodes": len(episodes), "time": time.time()})
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
