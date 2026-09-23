"""RGB choice scoring through vLLM's first-output-position logits."""

from math import exp


def _softmax(values):
    peak = max(values)
    weights = [exp(v - peak) for v in values]
    total = sum(weights)
    return [w / total for w in weights]


class VJev:
    def __init__(self, model, **engine_args):
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=model, **engine_args, generation_config="vllm",
            logprobs_mode="raw_logits", max_logprobs=-1,
        )
        self.params = SamplingParams(temperature=0, max_tokens=1, logprobs=-1)
        self.tokenizer = self.llm.get_tokenizer()

    def _token_id(self, label):
        ids = self.tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Output label must be one token: {label!r}")
        return ids[0]

    def select(self, image_url, prompt, choices, mode="choice_logits"):
        """choices maps single-token labels to descriptions; image_url may be a data URL."""
        if not choices:
            raise ValueError("choices must not be empty")
        if mode not in ("choice_logits", "binary_yes_no"):
            raise ValueError(f"Unknown mode: {mode}")
        labels = list(choices)
        ids = [self._token_id(label) for label in labels]
        if len(set(ids)) != len(ids):
            raise ValueError("Choice labels must have distinct token IDs")
        options = "\n".join(f"{key}: {value}" for key, value in choices.items())
        texts = [f"{prompt}\n\nOptions:\n{options}\nReply with one option label only."]
        if mode == "binary_yes_no":
            no, yes = self._token_id("no"), self._token_id("yes")
            texts += [
                f"{prompt}\n\nCandidate: {value}\n"
                "Is this the correct choice? Reply with yes or no only."
                for value in choices.values()
            ]
        requests = [[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": text},
        ]}] for text in texts]
        outputs = self.llm.chat(
            requests, self.params, use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False},
        )

        raw = []
        selected = []
        for i, output in enumerate(outputs):
            result = output.outputs[0]
            token_ids = ids if i == 0 else [no, yes]
            logits = [result.logprobs[0][t].logprob for t in token_ids]
            selected.append(logits)
            raw.append({
                "text": result.text, "token_ids": list(result.token_ids),
                "finish_reason": result.finish_reason,
                "prompt_token_ids": output.prompt_token_ids,
                "scored_token_ids": token_ids, "logits": logits,
            })
        scores = (_softmax(selected[0]) if mode == "choice_logits" else
                  [_softmax(pair)[1] for pair in selected[1:]])
        return {
            "mode": mode, "choice": labels[max(range(len(scores)), key=scores.__getitem__)],
            "scores": dict(zip(labels, scores)), "requests": requests,
            "native": raw[0], "candidate_outputs": raw[1:],
        }
