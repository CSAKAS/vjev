# VJev

输入 RGB、任务和选项，调用 vLLM 读取首位置 logits，返回得分最高的选项。

输入顺序：固定 system（任务、图像方向、输出格式）→ 当前 RGB → 选项/候选。同一任务保持 `prompt` 不变；缓存命中仍受引擎块大小限制。

依赖 `vllm==0.19.1`。在 `vjev` 的父目录（如 `~/Project`）运行：

```python
import base64
from pathlib import Path
from vjev import VJev

model = VJev("Qwen/Qwen3.5-9B", dtype="bfloat16")
rgb = base64.b64encode(Path("frame.png").read_bytes()).decode()
result = model.select(
    image_url=f"data:image/png;base64,{rgb}",
    prompt="选择让红色方块接近绿色终点的下一步移动。",
    choices={"A": "上", "B": "下", "C": "左", "D": "右"},
    mode="choice_logits",
)
print(result["choice"], result["scores"])
print(result["native"])  # 同底座原生 1-token 输出
```

- `choice_logits`：一条输入，比较 A/B/C/D 的 logits（默认）。
- `binary_yes_no`：每个选项分别判断 yes/no，比较 yes 分数，另附原生四选一对照。

`__init__.py` 是包入口，也包含全部实现。输出标签须是单 token，选项描述不限长度。实验时保留原生输出，并分别运行 VJev 和原生控制轨迹。
