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
        self.params = SamplingParams(temperature=0, max_tokens=1, logprobs=-1, detokenize=False)
        self.tokenizer = self.llm.get_tokenizer()

    def _token_id(self, label):
        ids = self.tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Output label must be one token: {label!r}")
        return ids[0]

    def select(self, image_url, prompt, choices, mode="choice_logits",
               order="system_image_question"):
        """prompt is the fixed task; choices maps single-token labels to descriptions."""
        if not choices:
            raise ValueError("choices must not be empty")
        if mode not in ("choice_logits", "binary_yes_no"):
            raise ValueError(f"Unknown mode: {mode}")
        if order not in ("system_image_question", "system_question_image", "image_rules_question"):
            raise ValueError(f"Unknown order: {order}")
        labels = list(choices)
        ids = [self._token_id(label) for label in labels]
        if len(set(ids)) != len(ids):
            raise ValueError("Choice labels must have distinct token IDs")
        system = (
            f"Task: {prompt}\n"
            "Use the image as evidence. Directions are relative to the image: "
            "up is toward the top, down toward the bottom, left and right as displayed.\n"
            "For an options question, reply with one option label only. "
            "For a candidate question, reply with lowercase yes or no only."
        )
        options = "\n".join(f"{key}: {value}" for key, value in choices.items())
        texts = [f"Choose the correct option.\nOptions:\n{options}"]
        if mode == "binary_yes_no":
            no, yes = self._token_id("no"), self._token_id("yes")
            texts += [
                f"Candidate: {value}\nIs this the correct choice?"
                for value in choices.values()
            ]
        requests = []
        for text in texts:
            image = {"type": "image_url", "image_url": {"url": image_url}}
            question = {"type": "text", "text": "\n\n" + text + "\n\n"}
            messages = [{"role": "system", "content": system}]
            content = [question, image] if order == "system_question_image" else [image, question]
            if order == "image_rules_question":
                messages = []
                content.insert(1, {"type": "text", "text": "\n\n" + system})
            requests.append(messages + [{"role": "user", "content": content}])
        outputs = self.llm.chat(
            requests, self.params, use_tqdm=False,
            chat_template_content_format="openai",
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
                "text": self.tokenizer.decode(result.token_ids, skip_special_tokens=True),
                "token_ids": list(result.token_ids),
                "finish_reason": result.finish_reason,
                "prompt_token_ids": output.prompt_token_ids,
                "scored_token_ids": token_ids, "logits": logits,
            })
        scores = (_softmax(selected[0]) if mode == "choice_logits" else
                  [_softmax(pair)[1] for pair in selected[1:]])
        return {
            "mode": mode, "order": order,
            "choice": labels[max(range(len(scores)), key=scores.__getitem__)],
            "scores": dict(zip(labels, scores)), "requests": requests,
            "native": raw[0], "candidate_outputs": raw[1:],
        }
