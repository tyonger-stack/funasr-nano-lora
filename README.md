# funasr-nano-lora

**Fun-ASR-Nano-2512 在 Apple Silicon 上的 MLX LoRA 微调工具**(河南话 ASR 实战配方)。

在 MacBook(M4 Max / 64GB)上对 FunAudioLLM/Fun-ASR-Nano-2512(800M,SenseVoice SAN-M 编码器 + Qwen3-0.6B 解码器)做 LoRA 微调:无需 CUDA,单卡训练一轮约 10 小时,800M 模型在河南话方言测试集上相对零样本降低 ~30% CER。

## 方法(最佳配方 E4)

移植 Qwen3-ASR 社区 allQKVO+Projector 配方到 funasr-mlx 推理实现之上,自建训练器:

| 要素 | 配置 |
|---|---|
| LoRA 目标 | 解码器注意力 **q/k/v/o 全 4 投影 ×28 层**(112 个)+ **audio_adaptor 投影器 linear1/linear2**(2 个,对应 Qwen3-ASR 的 multi_modal_projector) |
| r / α / scale | 16 / 16 / 1 |
| 其余参数 | 全部冻结(仅 LoRA 可训练,~4.68M 参数) |
| lr | 1e-4 cosine 退火到 0,warmup 600 步 |
| batch | 2 样本 × 16 梯度累积 = 32 样本/步,19,200 步 ≈ 1 epoch(61 万条) |
| 稳定性 | 梯度范数熔断(>100 跳过该步)+ clip 1.0 |
| 数据格式 | metadata.jsonl(`{"audio": 路径, "transcription": 文本}`),flac/wav 16k mono,0.5–30s |

三个关键工程决策(均经过消融验证):
1. **adaptor 投影器挂 LoRA,而不是整棵解冻** —— 整棵解冻(12.6M 参数)在两个冻结巨塔之间制造移动目标,训练震荡且无增益;LoRA 方式与解码器同 lr 协同更新,更稳。
2. **伪标注数据需谨慎** —— 外部模型生成的伪转录带"规整化"风格(儿化脱落、语气词漂移),2% 的污染数据实测拖累 1pt+ CER,且退步集中在风格细节密集的说话人。
3. **训练/推理 prompt 逐字一致** —— 沿用官方 ChatML(`语音转写，不进行文本规整：<|startofspeech|>…`),评测直接复用同一拼装函数。

## 实测结果(河南话方言 ASR,6,242 条说话人级隔离测试集)

| 配置 | 河南 CER | AISHELL CER | 备注 |
|---|---|---|---|
| 零样本 | 12.59% | 2.73%@300 | fp16 基座 |
| S1: q/v LoRA r16, 600 步(3% 数据) | 9.58% | 1.70% | 18 分钟 |
| **E1: q/v LoRA r16, 1 epoch** | **8.88%** | **1.63%** | 10.1h |
| E3: r32+整棵解冻 adaptor+伪标注数据 | 中止 | — | 伪标注风格污染,详见 README 下文 |
| **E4: allQKVO+Projector(E1 同数据)** | **8.75%** | **1.57%** | **本仓库推荐配方;12.9h,全程零梯度熔断** |

错误预算分解(E1 全量,指导下一步优化方向):近音混淆 35%(需解冻 encoder 顶层)/ 风格-儿化语气词 30%(需标注对齐,零训练成本)/ 同音字 16%(解码器 LM 能力)/ 其它 18%。

## 快速开始

```bash
# 1) 环境(依赖 funasr-mlx 推理库)
python3.12 -m venv .venv && source .venv/bin/activate
pip install mlx transformers numpy soundfile pyyaml huggingface_hub tqdm pypinyin matplotlib

git clone https://github.com/baicai-1145/funasr-mlx.git   # 放在同级或任意路径
export PYTHONPATH=/path/to/funasr-mlx/src

# 2) fp16 权重(LoRA 不可用 W4 量化基座)
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('baicai1145/fun-asr-nano-mlx', allow_patterns=['fp16/*'], local_dir='weights/fun-asr-nano-mlx')"

# 3) 三道 sanity 门(必过:前向等价/单样本过拟合/冻结核对)
python funasr-nano-lora/sanity.py --base weights/fun-asr-nano-mlx/fp16 \
    --data_file /path/to/eval/metadata.jsonl --n 20

# 4) 训练(E4 配方)
caffeinate -is python funasr-nano-lora/train.py --config funasr-nano-lora/configs/henan-aishell-nano-e4.yaml

# 5) 评测(CER 口径对齐 qwen3-asr-ft 的 src/cer.py)
python funasr-nano-lora/eval_cer.py --base weights/fun-asr-nano-mlx/fp16 \
    --adapter outputs/lora-nano/henan-aishell-e4/adapter-019200.safetensors \
    --data_file /path/to/eval/metadata.jsonl --tag e4 --limit 300

# 6) 折叠成独立部署模型(funasr-mlx 直接加载)
python funasr-nano-lora/merge.py --base weights/fun-asr-nano-mlx/fp16 \
    --adapter outputs/lora-nano/henan-aishell-e4/adapter-019200.safetensors \
    --out outputs/merged-e4

# 实时训练曲线
python funasr-nano-lora/plot_curve.py --run henan-aishell-e4
```

注:`eval_cer.py` 顶部 `Q3_REPO` 指向本机 qwen3-asr-ft 仓库(仅用其 `src/cer.py` 的 CER 归一化与编辑距离,两行函数,可自行内联);配置里 `base_dir` / `data_file` 按本机路径修改。

## 代码地图(共 ~1000 行,零框架依赖)

| 文件 | 职责 |
|---|---|
| `nano_common.py` | 官方 prompt 协议(ChatML)、fbank+LFR 前端、speech 嵌入拼接、KV-cache 贪心解码(逐行对齐 funasr-mlx 参考实现,20/20 逐 token 等价验证) |
| `lora.py` | `LoRALinear`、任意目标层注入、adaptor 投影器注入、路径式 adapter 保存/加载 |
| `train.py` | config 驱动训练循环:token 排序批、CE 只对 assistant 段、梯度累积、梯度熔断、断点续训、时长缓存 |
| `eval_cer.py` | 基座/adapter/merged 三种模式评测,字级 CER |
| `merge.py` | 按模块路径把 LoRA 折叠回基座(`W' = W + (α/r)·B@A`),产出自包含 checkpoint |
| `sanity.py` | 三道训练前验证门 |
| `plot_curve.py` | 实时 loss/lr/CER 曲线 |
| `configs/` | S1/E1/E2/E3/E4 全部实验配置(消融档案) |

## 实验档案(为什么是 E4)

- **S1→E1**:3% 数据(18 分钟)拿到大部分收益(9.58%),整 epoch 再赚 0.7pt(8.88%)——收益集中在 cosine 退火尾段。
- **E2**(第 2 个 epoch,峰值 lr 5e-5):S2 短实验已示无增益,中止。
- **E3**(r32 + 整棵解冻 adaptor + 郑州伪标注):失败。逐样本解剖发现退步 100% 集中于单一说话人的儿化/语气词风格漂移 —— 伪标注数据的"规整化"转录风格污染;整棵解冻 adaptor 造成两模块共适应震荡。
- **E4**(allQKVO + adaptor 投影器 LoRA + 去伪标注数据):同点位 @300 领先 E1 0.6pt,且全程零梯度熔断。

## 致谢

- [funasr-mlx](https://github.com/baicai-1145/funasr-mlx)(Apache-2.0):Fun-ASR-Nano 的 MLX 数值校验实现,本工具站在其模型/前端/解码之上,只新增训练侧
- [FunASR](https://github.com/modelscope/FunASR) 官方 LoRA recipe:超参语义参照(`examples/industrial_data_pretraining/fun_asr_nano/lora_finetune.sh`)
- [Qwen3-ASR 社区配方](https://github.com/QwenAudio):allQKVO+Projector 目标层选择
