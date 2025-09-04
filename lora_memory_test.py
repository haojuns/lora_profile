import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from typing import List, Optional
import threading
import time
import queue
from datetime import datetime

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from typing import List, Optional

# LoRA base layer class
class LoRALayer():
    def __init__(
        self, 
        r: int,                    # LoRA rank
        lora_alpha: int,           # Scaling factor
        lora_dropout: float,       # Dropout ratio
        merge_weights: bool,       # Flag for weight merging
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merged = False
        self.merge_weights = merge_weights

# LoRA linear layer implementation
class Linear(nn.Linear, LoRALayer):
    def __init__(
        self, 
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                          merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
            
            #添加
            self.bias.requires_grad = False
        
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
            
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
            
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, use_lora: bool = True, lora_r: int = 4, 
                 lora_alpha: int = 1, lora_dropout: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        # Replace nn.Linear with our LoRA Linear
        lora_config = dict(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout) if use_lora else dict(r=0)
        self.W_q = Linear(d_model, d_model, **lora_config)
        self.W_k = Linear(d_model, d_model, **lora_config)
        self.W_v = Linear(d_model, d_model, **lora_config)
        self.W_o = Linear(d_model, d_model, **lora_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Linear transformations with LoRA
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attn = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Output projection with LoRA
        output = self.W_o(context)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, use_lora: bool = True, 
                 lora_r: int = 4, lora_alpha: int = 1, lora_dropout: float = 0.):
        super().__init__()
        self.attention = MultiheadAttention(d_model, num_heads, use_lora, lora_r, lora_alpha, lora_dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network with optional LoRA
        lora_config = dict(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout) if use_lora else dict(r=0)
        self.feed_forward = nn.Sequential(
            Linear(d_model, d_ff, **lora_config),
            nn.ReLU(),
            Linear(d_ff, d_model, **lora_config)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x

class Model(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, use_lora: bool = True,
                 lora_r: int = 4, lora_alpha: int = 1, lora_dropout: float = 0.):
        super().__init__()
        self.block1 = TransformerBlock(d_model, num_heads, d_ff, use_lora, lora_r, lora_alpha, lora_dropout)
        self.block2 = TransformerBlock(d_model, num_heads, d_ff, use_lora, lora_r, lora_alpha, lora_dropout)
        self.output_layer = Linear(d_model, d_model, r=lora_r if use_lora else 0, 
                                 lora_alpha=lora_alpha, lora_dropout=lora_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return self.output_layer(x)



def freeze_parameters(model: nn.Module, exclude_patterns: Optional[List[str]] = None):
    """冻结模型参数，可以指定排除的参数名模式"""
    if exclude_patterns is None:
        exclude_patterns = []
    
    has_trainable = False
    for name, param in model.named_parameters():
        should_freeze = True
        for pattern in exclude_patterns:
            # 如果条件是字符串，则检查该字符串是否在参数名中
            if isinstance(pattern, str) and pattern in name:
                should_freeze = False
                has_trainable = True
                break
            # 如果条件是正则表达式，则进行正则匹配
            elif isinstance(pattern, str) and re.match(pattern, name):
                should_freeze = False
                has_trainable = True
                break
        param.requires_grad = not should_freeze
        
    
    
    # 如果没有可训练的参数，至少保持一个参数可训练
    if not has_trainable:
        first_param = list(model.parameters())[-1]
        first_param.requires_grad = True

class MemoryMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.memory_queue = queue.Queue()
        self.stop_flag = threading.Event()
        self.start_time = None
        
    def monitor_memory(self):
        self.start_time = time.time()
        while not self.stop_flag.is_set():
            memory = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
            current_time = time.time() - self.start_time
            self.memory_queue.put((current_time, memory))
            time.sleep(self.interval)
    
    def start(self):
        self.stop_flag.clear()
        self.monitor_thread = threading.Thread(target=self.monitor_memory)
        self.monitor_thread.start()
    
    def stop(self):
        self.stop_flag.set()
        self.monitor_thread.join()
    
    def get_measurements(self):
        measurements = []
        while not self.memory_queue.empty():
            measurements.append(self.memory_queue.get())
        return sorted(measurements)  # Sort by timestamp

def train_and_monitor_memory(
    model: nn.Module,
    input_data: torch.Tensor,
    target: torch.Tensor,
    num_steps: int,
    freeze_pattern: Optional[List[str]] = None
) -> List[tuple]:
    """训练模型并监控内存使用情况"""
    
    if freeze_pattern is not None:
        for param in model.parameters():
            param.requires_grad = False
        freeze_parameters(model, freeze_pattern)
    else:
        for param in model.parameters():
            param.requires_grad = True
    
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)
    criterion = nn.MSELoss()
    
    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    torch.cuda.empty_cache()
    
    # 创建内存监视器
    monitor = MemoryMonitor(interval=0.0000001)
    monitor.start()
    
    # 打印可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Trainable parameters: {trainable_params:,} / {total_params:,}')
    
    try:
        for step in range(num_steps):
            optimizer.zero_grad()
            output = model(input_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if (step + 1) % 5 == 0:
                print(f'Step {step+1}, Loss: {loss.item():.4f}')
    
    finally:
        # 确保停止监视器
        monitor.stop()
    
    torch.cuda.empty_cache()
    
    # 获取所有测量结果
    measurements = monitor.get_measurements()
    return measurements

def run_experiment():
    # 设置随机种子
    torch.manual_seed(42)
    
    # 模型参数
    d_model = 256
    num_heads = 8
    d_ff = 1024
    batch_size = 32
    seq_length = 50
    num_steps = 2
    
    # 生成随机数据
    input_data = torch.randn(batch_size, seq_length, d_model).cuda()
    target = torch.randn(batch_size, seq_length, d_model).cuda()
    
    # 实验条件
    conditions = [
        ("All Frozen", []),
        ("Only LoRA", ["lora"]),
        ("Only Block1 LoRA", [r"block1.*lora"]),
        ("Only Block2 LoRA", [r"block2.*lora"]),
        ("All Trainable", None)
    ]
    
    # 运行实验并收集数据
    results = {}
    for condition_name, freeze_pattern in conditions:
        print(f"\nRunning experiment: {condition_name}")
        model = Model(d_model, num_heads, d_ff).cuda()
        measurements = train_and_monitor_memory(model, input_data, target, num_steps, freeze_pattern)
        results[condition_name] = measurements
        
        # 清理GPU内存
        del model
        torch.cuda.empty_cache()
        time.sleep(1)  # 给系统一些时间来清理内存
    
    return results

if __name__ == "__main__":
    results = run_experiment()
    
    # 绘制结果
    plt.figure(figsize=(12, 6))
    for condition_name, measurements in results.items():
        times, memories = zip(*measurements)
        plt.plot(times, memories, label=condition_name)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Real-time Memory Usage During Training with Different Parameter Freezing Strategies')
    plt.legend()
    plt.grid(True)
    
    # 保存图表时添加时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig('TaskMate/taskmate/playground/memory_usage.png')
    plt.close()