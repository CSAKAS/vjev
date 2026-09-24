# VJev

输入 RGB、任务和选项，调用 vLLM 读取首位置 logits，返回得分最高的选项。

默认顺序：固定 system（任务、图像方向、输出格式）→ 当前 RGB → 选项/候选。同一任务保持 `prompt` 不变；缓存命中仍受引擎块大小限制。

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

`order` 可选 `system_image_question`（默认）、`system_question_image`、`image_rules_question`（规则放入 user）。图文顺序显式保留；原生输出由实际生成的 token 还原。

`__init__.py` 是包入口，也包含全部实现。输出标签须是单 token，选项描述不限长度。实验时保留原生输出，并分别运行 VJev 和原生控制轨迹。

小规模验证：Qwen3.5-9B，4×A800，纯 RGB 输入，默认输入顺序；每个游戏 50 个场景，各控制器独立运行。

| 任务成功次数 / 50 | 四选一 logits | 候选 yes/no | 原生 1-token |
|---|---:|---:|---:|
| 无障碍导航 | 41 | 41 | 41 |
| 绕墙导航 | 0 | 1 | 0 |
| 颜色匹配 | 50 | 49 | 50 |
| 数量比较 | 45 | 46 | 44 |

四选一与原生读取同一次推理结果；最高分并列时的处理不同可能导致答案不同。本轮每次提交四选一和四个 yes/no 请求，仅比较准确率，未验证缓存收益。完整实验脚本和数据留存本地，不纳入此仓库。
