"""Audit saved calls/independent trajectories and produce a compact report."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path

from games import ACTIONS, Game
from order import CONTROLLERS, ORDERS

NAMES = {"navigation": "无障碍导航", "detour": "绕墙导航",
         "color_match": "颜色匹配", "count_max": "数量比较"}
ORDER_NAMES = dict(zip(ORDERS, ["system→RGB→问题", "system→问题→RGB", "RGB→规则→问题†"]))


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    root = p.parse_args().root
    all_metrics, audit, differences = {}, {}, []
    for game, name in NAMES.items():
        folder = root / game
        assert read(folder / "completion.json")["complete"]
        settings = read(folder / "settings.json")
        assert settings["vllm"] == "0.19.1"
        assert settings["dtype"] == "BF16" and settings["adapter"] is None
        assert not settings["thinking"]
        manifest = settings["manifest"]
        assert len(manifest) == 50
        assert len({hashlib.sha256(Game(game, c["seed"]).render()).hexdigest() for c in manifest}) == 50
        calls = {}
        forwards = decode = invalid = disagreements = ties = 0
        for path in sorted((folder / "calls").glob("*.json")):
            rec = read(path)
            result = rec["result"]
            rgb = (folder / "images" / (rec["image_sha256"] + ".png")).read_bytes()
            assert hashlib.sha256(rgb).hexdigest() == rec["image_sha256"]
            assert len(result["requests"]) == 5
            rule = None
            for i, request in enumerate(result["requests"]):
                content = request[-1]["content"]
                parts = [x for x in content if x["type"] == "image_url"]
                assert len(parts) == 1
                assert base64.b64decode(parts[0]["image_url"]["url"].split(",", 1)[1]) == rgb
                if rec["order"] != ORDERS[2]:
                    assert request[0]["role"] == "system"
                    current_rule = request[0]["content"]
                else:
                    assert len(request) == 1
                    current_rule = content[1]["text"].strip()
                if rule is None: rule = current_rule
                assert current_rule == rule
                assert settings["task"] in rule
            for raw, label in zip(result["candidate_outputs"], ACTIONS):
                no, yes = raw["logits"]
                peak = max(no, yes)
                expected = math.exp(yes - peak) / (math.exp(no - peak) + math.exp(yes - peak))
                assert math.isclose(result["scores"][label], expected, abs_tol=1e-9)
                assert len(raw["token_ids"]) == 1
            assert all(math.isfinite(x) for x in result["scores"].values())
            assert rec["predictions"]["binary_yes_no"] == max(ACTIONS, key=result["scores"].__getitem__)
            assert rec["predictions"]["choice_logits"] == list(ACTIONS)[max(range(4), key=result["native"]["logits"].__getitem__)]
            assert len(result["native"]["token_ids"]) == 1
            native_label = dict(zip(result["native"]["scored_token_ids"], ACTIONS)).get(result["native"]["token_ids"][0])
            assert rec["predictions"]["native"] == native_label
            invalid += native_label is None
            if native_label != rec["predictions"]["choice_logits"]:
                disagreements += 1
                if native_label is not None:
                    logits = dict(zip(ACTIONS, result["native"]["logits"]))
                    ties += logits[native_label] == max(logits.values())
            assert all(row["decode_tokens"] == 0 for row in rec["trace"])
            forwards += sum(row["model_forward_calls"] for row in rec["trace"])
            decode += sum(row["decode_tokens"] for row in rec["trace"])
            calls[rec["id"]] = rec
        initial = read(folder / "initial.json")
        assert len(initial) == 150
        for row in initial:
            g = Game(game, manifest[row["case"]]["seed"])
            assert calls[row["call"]]["image_sha256"] == hashlib.sha256(g.render()).hexdigest()
            assert row["optimal"] == g.optimal()
        episodes = [json.loads(line) for line in (folder / "episodes.jsonl").read_text().splitlines()]
        assert len(episodes) == 450
        assert len({(e["case"]["index"], e["order"], e["controller"]) for e in episodes}) == 450
        for episode in episodes:
            g = Game(game, episode["case"]["seed"])
            for step in episode["steps"]:
                rec = calls[step["call"]]
                assert rec["order"] == episode["order"]
                assert rec["image_sha256"] == hashlib.sha256(g.render()).hexdigest()
                assert step["action"] == rec["predictions"][episode["controller"]]
                assert step["state"] == g.state() and step["optimal"] == g.optimal()
                if step["action"] is not None:
                    assert step["execution"] == g.step(step["action"])
            assert g.success == episode["success"]
        metrics = read(folder / "metrics.json")
        for order in ORDERS:
            for controller in CONTROLLERS:
                group = [e for e in episodes if e["order"] == order and e["controller"] == controller]
                assert len(group) == 50
                m = metrics[order][controller]
                assert m["success"] == sum(e["success"] for e in group)
                assert m["initial_correct"] == sum(r["predictions"][controller] in r["optimal"] for r in initial if r["order"] == order)
        for case in range(50):
            rows = {r["order"]: r for r in initial if r["case"] == case}
            for controller in CONTROLLERS:
                predictions = [rows[o]["predictions"][controller] for o in ORDERS]
                if len(set(predictions)) > 1:
                    differences.append({"game": game, "case": case, "controller": controller,
                                        "predictions": dict(zip(ORDERS, predictions)),
                                        "optimal": rows[ORDERS[0]]["optimal"],
                                        "calls": {o: rows[o]["call"] for o in ORDERS}})
        all_metrics[game] = metrics
        audit[game] = {"unique_calls": len(calls), "logical_requests": len(calls)*5,
                       "episodes": len(episodes), "model_forward_calls": forwards,
                       "decode_tokens": decode, "http_requests": 0,
                       "native_invalid_calls": invalid, "native_choice_disagreements": disagreements,
                       "native_choice_tie_disagreements": ties}
    (root / "audit.json").write_text(json.dumps(audit, indent=2))
    (root / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    (root / "differences.json").write_text(json.dumps(differences, indent=2))
    totals = {o: {c: sum(m[o][c]["success"] for m in all_metrics.values())
                  for c in CONTROLLERS} for o in ORDERS}
    winners = {max(ORDERS, key=lambda o: totals[o][c]) for c in CONTROLLERS}
    conclusion = (f"本轮三个读出的总成功次数最多的顺序都是 **{ORDER_NAMES[next(iter(winners))]}**。"
                  if len(winners) == 1 else "不同读出的最佳顺序不同，见下表。")
    lines = ["# 输入顺序准确率对比", "", conclusion, "",
             "Qwen3.5-9B（无 LoRA），BF16 权重、FP32 递归状态、thinking=false；4×A800，每卡一个游戏。每游戏 50 个相同初始 RGB，共 1,800 条独立控制轨迹。实际调用 `vjev.VJev.select`。", "",
             "单元格为 **四选一 logits / 候选 yes-no / 原生 greedy 1-token** 的正确或成功次数，分母均为 50。", ""]
    for title, key in [("首帧动作准确率", "initial_correct"), ("独立任务成功率", "success")]:
        lines += [f"## {title}", "", "| 游戏 | system→RGB→问题 | system→问题→RGB | RGB→规则→问题† |",
                  "|---|---:|---:|---:|"]
        for game, name in NAMES.items():
            cells = [" / ".join(str(all_metrics[game][o][c][key]) for c in CONTROLLERS) for o in ORDERS]
            lines.append("| " + " | ".join([name, *cells]) + " |")
        lines.append("")
    lines += ["## 边界", "",
              "- 前两组只调整同一 user 消息中的图文顺序；†第三组还把完全相同的规则从 system 移到 user，不能单独归因于位置。",
              "- 显式保留 OpenAI 图文内容顺序；每条实际输入 token 均与模板展开加图像 token 扩展核对一致。模型输入不含环境状态、历史、正确答案或候选过滤。",
              "- 每次调用同时提交原生四选一和四个 yes/no 请求；四选一 logits 与原生输出来自同一次 forward。所有输出无词表约束，保留无效原生 token，不做修复。",
              "- 四选一最高分并列时按 UP、DOWN、LEFT、RIGHT 的插入顺序选取；原生由引擎全词表 argmax 决定，可能因并列处理不同而出现不同答案。",
              "- 每种控制器独立推进环境；相同 RGB/顺序调用复用已保存的模型结果，只用于准确率，不作速度比较。导航上限 20 步，连续 3 次无位移判停。",
              "- 单帧导航可能有多个正确方向，按是否缩短最短路径判分；闭环成功要求真正到达终点。分数不是成功概率。",
              "", f"审计：{sum(a['unique_calls'] for a in audit.values())} 次模型调用、"
              f"{sum(a['model_forward_calls'] for a in audit.values())} 次模型 forward，0 decode、0 HTTP。"
              f"原生候选外输出 {sum(a['native_invalid_calls'] for a in audit.values())} 次；"
              f"与四选一分歧 {sum(a['native_choice_disagreements'] for a in audit.values())} 次，"
              f"其中并列分数 {sum(a['native_choice_tie_disagreements'] for a in audit.values())} 次。",
              "", "原始数据：各游戏 `calls/`、`images/`、`initial.json`、`episodes.jsonl`；审计见 `audit.json`，分歧样例见 `differences.json`。", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
