import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union

import torch
import transformers
import sklearn
import numpy as np
from torch.utils.data import Dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

# 定义模型相关的命令行参数
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: bool = field(default=False, metadata={"help": "是否使用 LoRA 技术进行高效微调"})
    lora_r: int = field(default=8, metadata={"help": "LoRA 的秩 (Rank)"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA 的缩放因子 Alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA 层的 Dropout 率"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "指定要在哪些模块应用 LoRA (如 query,value)"})


# 定义数据相关的命令行参数
@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "训练数据所在的文件夹路径"})
    kmer: int = field(default=-1, metadata={"help": "输入序列的 k-mer 长度。-1 表示不使用 k-mer (适用于 DNABERT-2)"})


# 定义训练相关的参数，继承自 transformers.TrainingArguments
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "序列的最大长度"})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps"),
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """安全地收集状态字典并保存到磁盘，处理 CPU/GPU 转换"""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def get_alter_of_dna_sequence(sequence: str):
    """获取 DNA 序列的转换版本（当前实现仅为补码）"""
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    # return "".join([MAP[c] for c in reversed(sequence)]) # 如果需要反向补码则取消此行注释
    return "".join([MAP[c] for c in sequence])

def generate_kmer_str(sequence: str, k: int) -> str:
    """将 DNA 序列转换为以空格分隔的 k-mer 字符串"""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """加载已有的 k-mer JSON 文件或生成新的。生成的 JSON 会保存在原数据同目录下"""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"从 {kmer_path} 加载 k-mer...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:        
        logging.warning(f"正在生成 k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"保存 k-mer 到 {kmer_path}...")
            json.dump(kmer, f)
        
    return kmer

class SupervisedDataset(Dataset):
    """用于监督式微调的数据集类"""

    def __init__(self, 
                 data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 kmer: int = -1):

        super(SupervisedDataset, self).__init__()

        # 从磁盘读取 CSV 数据
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]

        # 根据列数判断是单序列分类还是序列对分类
        if len(data[0]) == 2:
            # 数据格式: [text, label]
            logging.warning("执行单序列分类任务...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # 数据格式: [text1, text2, label]
            logging.warning("执行序列对分类任务...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            raise ValueError("不支持的数据格式。")
        
        # 如果需要 k-mer 预处理
        if kmer != -1:
            # 确保在分布式训练时，只有主进程进行文件写操作
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"使用 {kmer}-mer 作为输入...")
            texts = load_or_generate_kmer(data_path, texts, kmer)

            if torch.distributed.get_rank() == 0:
                torch.distributed.barrier()

        # 调用分词器对文本进行编码
        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest", # 填充到最长长度
            max_length=tokenizer.model_max_length,
            truncation=True, # 超出部分截断
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = len(set(labels)) # 自动计算类别数

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """用于 Batch 拼接的数据处理器，负责动态 Padding"""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # 将 input_ids 填充为 Tensor
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id), # 生成 Attention Mask
        )

def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
    """使用 sklearn 手动计算评价指标：准确率、F1、MCC、精准率、召回率"""
    valid_mask = labels != -100  # 排除填充标记
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }

def preprocess_logits_for_metrics(logits:Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    """在计算指标前预处理 Logits，只保留预测的类别索引以节省显存"""
    if isinstance(logits, tuple):
        logits = logits[0]

    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])

    return torch.argmax(logits, dim=-1)


def compute_metrics(eval_pred):
    """Huggingface Trainer 调用的评价指标计算接口"""
    predictions, labels = eval_pred
    return calculate_metric_with_sklearn(predictions, labels)



def train():
    # 解析命令行参数
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 加载分词器 (Tokenizer)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True, # 允许加载 DNABERT-2 的自定义分词逻辑
    )

    # 特殊处理 InstaDeepAI 的模型
    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token

    # 定义训练、验证和测试数据集
    train_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                      data_path=os.path.join(data_args.data_path, "train.csv"), 
                                      kmer=data_args.kmer)
    val_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=os.path.join(data_args.data_path, "dev.csv"), 
                                     kmer=data_args.kmer)
    test_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=os.path.join(data_args.data_path, "test.csv"), 
                                     kmer=data_args.kmer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)


    # 加载用于序列分类的模型
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        num_labels=train_dataset.num_labels, # 自动配置输出层的类别数
        trust_remote_code=True,
    )

    # 如果启用 LoRA 技术
    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=list(model_args.lora_target_modules.split(",")),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS", # 序列分类任务类型
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters() # 打印可训练参数量

    # 定义 Huggingface Trainer
    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                                   compute_metrics=compute_metrics,
                                   train_dataset=train_dataset,
                                   eval_dataset=val_dataset,
                                   data_collator=data_collator)
    # 开始训练
    trainer.train()

    # 如果设置了保存模型
    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # 训练结束后，在测试集上评估结果并保存
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    train()
