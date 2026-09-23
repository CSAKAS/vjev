# 复现顺序实验

每个游戏 50 个场景，三种顺序、三种独立控制器。输入只有 RGB 和固定任务文本。

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/run.sh --model /path/to/Qwen3.5-9B --game navigation --n 50 --output /tmp/order-test/navigation
```

同样运行 `detour`、`color_match`、`count_max`，输出到同名子目录；四组结束后运行 `python experiments/report_order.py /tmp/order-test` 审计并生成报告。

核心 `VJev.select` 使用相同文字，仅调整顺序。第三组同时改变规则消息角色。原生 MCQ 和四个 yes/no 请求一起提交；四选一读出取同次 MCQ 的 logits。相同 RGB/顺序的结果用于各自独立轨迹，本实验不比较速度。
