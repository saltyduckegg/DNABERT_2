# DNABERT-2 微调脚本 (train.py) 详细解读

本文件提供了 `finetune/train.py` 脚本的深度解读，旨在帮助开发者理解其架构、核心技术并能够快速应用于自己的数据集。

## 1. 脚本概览

`train.py` 是一个基于 Hugging Face `transformers` 库构建的 DNA 序列分类微调框架。它支持：
*   **多模型兼容**：支持 DNABERT-2、传统的 k-mer DNABERT 以及 Nucleotide Transformer。
*   **参数高效微调 (PEFT)**：集成了 LoRA 技术，允许在有限显存下微调超大规模模型。
*   **定制化数据处理**：支持单序列和序列对分类，具备自动 k-mer 生成和缓存功能。

---

## 2. 核心架构设计

### 2.1 参数管理
脚本通过三个数据类（Dataclasses）管理参数：
1.  **ModelArguments**: 处理模型路径及 LoRA 配置（秩、Alpha、目标模块）。
2.  **DataArguments**: 指定数据路径和 k-mer 长度（DNABERT-2 设为 -1）。
3.  **TrainingArguments**: 继承自标准训练参数，控制学习率、Batch Size、训练轮数等。

### 2.2 数据处理流程 (`SupervisedDataset`)
*   **格式要求**：CSV 文件，首行为表头。
    *   2 列：`sequence, label` (单序列分类)
    *   3 列：`sequence1, sequence2, label` (序列对分类)
*   **分词策略**：
    *   **DNABERT-2**：使用自定义 BPE 分词，通过 `trust_remote_code=True` 加载。
    *   **DNABERT-1**：通过 `kmer` 参数自动将序列转换为以空格分隔的片段。
*   **动态填充 (`DataCollator`)**：仅在每个 Batch 内部进行 Padding，显著优化了长短不一序列的计算效率。

### 2.3 指标评估
脚本使用 `sklearn` 计算多项关键指标：
*   **Accuracy** (准确率)
*   **F1 (Macro)** (宏平均 F1)
*   **MCC (Matthews Correlation Coefficient)**：在类别不平衡的基因组任务中，这是最客观的指标。

---

## 3. 核心技术详解

### 3.1 LoRA 高效微调
对于参数量巨大的模型（如 2.5B 的 Nucleotide Transformer），全量微调不可行。LoRA 通过在 Transformer 的注意力层（通常是 Query 和 Value）注入可训练的低秩矩阵，仅训练约 1% 的参数即可达到接近全量微调的效果。

### 3.2 显存优化策略
*   **`preprocess_logits_for_metrics`**：在验证阶段，脚本会先在 GPU 上执行 `argmax` 操作。这意味着 Trainer 只需传输类别标签到内存，而不需要传输庞大的概率分布矩阵（Logits），有效防止显存溢出（OOM）。
*   **混合精度 (fp16)**：通过半精度浮点数减少权重和梯度的内存占用。

---

## 4. 使用指南与示例

### 4.1 数据准备
在同一目录下准备 `train.csv`, `dev.csv`, `test.csv`。

### 4.2 运行示例

#### 场景 A：微调 DNABERT-2 (标准模式)
```bash
python train.py \
    --model_name_or_path zhihan1996/DNABERT-2-117M \
    --data_path ./my_data \
    --kmer -1 \
    --model_max_length 128 \
    --per_device_train_batch_size 8 \
    --learning_rate 3e-5 \
    --num_train_epochs 5 \
    --fp16 \
    --output_dir ./output/dnabert2_finetuned
```

#### 场景 B：使用 LoRA 微调超大模型
```bash
python train.py \
    --model_name_or_path InstaDeepAI/nucleotide-transformer-2.5b-multi-species \
    --use_lora True \
    --lora_target_modules "query,value,key,dense" \
    --data_path ./my_data \
    --kmer -1 \
    --output_dir ./output/nt_lora
```

---

## 5. 开发者建议

1.  **序列长度设置**：DNABERT-2 建议 `--model_max_length` 设置为物理长度的 1/4 到 1/5。
2.  **分布式训练**：如果有多张 GPU，建议使用 `torchrun --nproc_per_node=N train.py`。
3.  **结果查看**：最终测试集表现会记录在 `output_dir/results/<run_name>/eval_results.json` 中。
