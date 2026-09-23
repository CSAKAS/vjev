# VJev

我们处理 RGB、任务和候选 → vLLM 执行 prefill → 返回首个输出位置的原始 logits → 我们计算分数和 top1。

一个 Python 文件，无引擎插件、padding 或自定义缓存逻辑。面向 vLLM **0.19.1**，使用其 Python 接口；模型须支持 RGB 和 chat template。

## 使用

在 `vjev` 目录的父目录运行，例如 `cd ~/Project`（推理环境需安装 `vllm==0.19.1`）：

```python
import base64
import json
from pathlib import Path
from vjev import VJev

model = VJev("Qwen/Qwen3.5-9B", dtype="bfloat16")
rgb = base64.b64encode(Path("frame.png").read_bytes()).decode()
result = model.select(
    image_url=f"data:image/png;base64,{rgb}",
    prompt="根据图像，选择红色方块接近绿色终点的下一步。方向按图像上下左右。",
    choices={"A": "向上移动", "B": "向下移动", "C": "向左移动", "D": "向右移动"},
    mode="choice_logits",  # 或 binary_yes_no
)
print(result["choice"], result["scores"])
Path("result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
```

仅 RGB 和上述任务/选项文本输入模型，不读取环境状态。选项描述可以多 token，输出标签必须是不同的单 token；不符合时直接报错。

## 两种读出

| mode | 内部请求 | 分数 |
|---|---|---|
| `choice_logits`（默认） | 一条含全部选项的输入 | 对选项标签 logits 做 softmax，再取 top1 |
| `binary_yes_no` | 每候选一条判断输入，另加一条四选一原生对照 | 每路对 `[no, yes]` 做 softmax，比较 yes 分数 |

所有请求均为无词表约束的 greedy `max_tokens=1`。原生输出保留在 `native`；yes/no 请求生成的实际 token 也保留在 `candidate_outputs`，选择只使用 logits，不解析或修补生成文本。并列时取选项插入顺序中的第一项。

`logprobs_mode="raw_logits"` 使返回对象的 `logprob` 字段实际承载原始 logit。`logprobs=-1` 获取全词表，避免所需标签不在 top-k 中；返回给调用者的结果只保留相关标签分数。该方案仍有全词表回传开销，尚未优化。

获取的是 prefill 末尾用于预测第一个输出 token 的分布，不是输入各位置的 `prompt_logprobs`。引擎仍执行一次采样；生成的 token 不再送回模型做后续 decode。prefill 可能被调度器分块，不能把一次调用视为一次 forward。

四候选 binary 模式一次 `chat` 提交 **5 条逻辑请求**（含原生对照），不保证一次 forward，也不保证公共前缀缓存命中。当前不提供延迟对比，不能把整批耗时当作单个方法的耗时。

这些分数不是任务成功概率；binary 各分数也不要求加起来等于 1。后续实验须使用同一底座、相同初始 RGB 和任务，为 VJev 和原生输出运行独立控制轨迹，保留输入、原始输出、分数和模型配置；不能把另一条轨迹上的旁路预测计为成功率。当前只提供调用封装，未进行新的 GPU 效果或性能实验。
