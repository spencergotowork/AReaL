from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist


@torch.compile
def _gather_logprobs(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0
):
    log_probs = torch.nn.functional.log_softmax(logits.float() / temperature, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_probs_labels


@torch.compile
def _gather_logprobs_entropy(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0
):
    log_probs = torch.nn.functional.log_softmax(logits.float() / temperature, dim=-1)
    entropy = -torch.sum(log_probs.exp() * log_probs, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_probs_labels, entropy


def gather_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 1024,
):
    batch_size = logits.shape[0]

    if batch_size <= chunk_size:
        return _gather_logprobs(logits, labels, temperature)

    log_probs_labels_list = []

    for i in range(0, batch_size, chunk_size):
        end_idx = min(i + chunk_size, batch_size)
        chunk_logits = logits[i:end_idx]
        chunk_labels = labels[i:end_idx]

        chunk_log_probs = _gather_logprobs(chunk_logits, chunk_labels, temperature)

        log_probs_labels_list.append(chunk_log_probs)

    return torch.cat(log_probs_labels_list)


class MemoryEfficientGatherLogprobs(torch.autograd.Function):
    """内存高效的gather_logprobs实现，在backward阶段也进行分块处理"""
    
    @staticmethod
    def forward(ctx, logits, labels, temperature, chunk_size):
        batch_size = logits.shape[0]
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size
        ctx.batch_size = batch_size
        
        if batch_size <= chunk_size:
            # 小批次直接处理
            log_probs = torch.nn.functional.log_softmax(logits.float() / temperature, dim=-1)
            log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            ctx.save_for_backward(logits, labels, log_probs)
            return log_probs_labels
        
        # 分块处理forward
        log_probs_labels_list = []
        chunk_info = []  # 存储每个chunk的信息用于backward
        
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            chunk_logits = logits[i:end_idx]
            chunk_labels = labels[i:end_idx]
            
            chunk_log_probs = torch.nn.functional.log_softmax(chunk_logits.float() / temperature, dim=-1)
            chunk_log_probs_labels = chunk_log_probs.gather(dim=-1, index=chunk_labels.unsqueeze(-1)).squeeze(-1)
            
            log_probs_labels_list.append(chunk_log_probs_labels)
            chunk_info.append((i, end_idx, chunk_logits, chunk_labels, chunk_log_probs))
        
        ctx.chunk_info = chunk_info
        return torch.cat(log_probs_labels_list)
    
    @staticmethod
    def backward(ctx, grad_output):
        logits, labels, log_probs = None, None, None
        temperature = ctx.temperature
        chunk_size = ctx.chunk_size
        batch_size = ctx.batch_size
        
        if batch_size <= chunk_size:
            # 小批次的backward
            logits, labels, log_probs = ctx.saved_tensors
            grad_logits = torch.zeros_like(logits)
            
            # 计算log_softmax的梯度
            grad_log_probs = torch.zeros_like(log_probs)
            grad_log_probs.scatter_add_(dim=-1, index=labels.unsqueeze(-1), src=grad_output.unsqueeze(-1))
            
            # 通过log_softmax反向传播到logits
            probs = log_probs.exp()
            grad_logits = grad_log_probs - probs * grad_log_probs.sum(dim=-1, keepdim=True)
            grad_logits = grad_logits / temperature
            
            return grad_logits, None, None, None
        
        # 分块处理backward
        # 创建完整的grad_logits tensor
        first_chunk_logits = ctx.chunk_info[0][2]
        grad_logits = torch.zeros_like(first_chunk_logits)
        if len(ctx.chunk_info) > 1:
            # 计算完整batch的大小
            total_size = sum(chunk_end - chunk_start for chunk_start, chunk_end, _, _, _ in ctx.chunk_info)
            grad_logits = torch.zeros(total_size, *first_chunk_logits.shape[1:], device=first_chunk_logits.device, dtype=first_chunk_logits.dtype)
        
        start_idx = 0
        for i, (chunk_start, chunk_end, chunk_logits, chunk_labels, chunk_log_probs) in enumerate(ctx.chunk_info):
            chunk_size = chunk_end - chunk_start
            chunk_grad_output = grad_output[start_idx:start_idx + chunk_size]
            
            # 计算当前chunk的梯度
            grad_log_probs = torch.zeros_like(chunk_log_probs)
            grad_log_probs.scatter_add_(dim=-1, index=chunk_labels.unsqueeze(-1), src=chunk_grad_output.unsqueeze(-1))
            
            # 通过log_softmax反向传播到logits
            probs = chunk_log_probs.exp()
            chunk_grad_logits = grad_log_probs - probs * grad_log_probs.sum(dim=-1, keepdim=True)
            chunk_grad_logits = chunk_grad_logits / temperature
            
            # 将chunk梯度放到正确的位置
            grad_logits[chunk_start:chunk_end] = chunk_grad_logits
            start_idx += chunk_size
        
        return grad_logits, None, None, None


def gather_logprobs_memory_efficient(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 1024,
):
    """
    内存高效的gather_logprobs实现，在backward阶段也进行分块处理
    
    Args:
        logits: 模型输出的logits [batch_size, seq_len, vocab_size]
        labels: 目标标签 [batch_size, seq_len]
        temperature: 温度参数
        chunk_size: 分块大小
        
    Returns:
        log_probs_labels: 对应的log概率 [batch_size, seq_len]
    """
    return MemoryEfficientGatherLogprobs.apply(logits, labels, temperature, chunk_size)


class MemoryEfficientGatherLogprobsEntropy(torch.autograd.Function):
    """内存高效的gather_logprobs_entropy实现，在backward阶段也进行分块处理"""
    
    @staticmethod
    def forward(ctx, logits, labels, temperature, chunk_size):
        batch_size = logits.shape[0]
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size
        ctx.batch_size = batch_size
        
        if batch_size <= chunk_size:
            # 小批次直接处理
            log_probs = torch.nn.functional.log_softmax(logits.float() / temperature, dim=-1)
            entropy = -torch.sum(log_probs.exp() * log_probs, dim=-1)
            log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            ctx.save_for_backward(logits, labels, log_probs)
            return log_probs_labels, entropy
        
        # 分块处理forward
        log_probs_labels_list = []
        entropy_list = []
        chunk_info = []  # 存储每个chunk的信息用于backward
        
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            chunk_logits = logits[i:end_idx]
            chunk_labels = labels[i:end_idx]
            
            chunk_log_probs = torch.nn.functional.log_softmax(chunk_logits.float() / temperature, dim=-1)
            chunk_entropy = -torch.sum(chunk_log_probs.exp() * chunk_log_probs, dim=-1)
            chunk_log_probs_labels = chunk_log_probs.gather(dim=-1, index=chunk_labels.unsqueeze(-1)).squeeze(-1)
            
            log_probs_labels_list.append(chunk_log_probs_labels)
            entropy_list.append(chunk_entropy)
            chunk_info.append((i, end_idx, chunk_logits, chunk_labels, chunk_log_probs))
        
        ctx.chunk_info = chunk_info
        return torch.cat(log_probs_labels_list), torch.cat(entropy_list)
    
    @staticmethod
    def backward(ctx, grad_output_logprobs, grad_output_entropy):
        logits, labels, log_probs = None, None, None
        temperature = ctx.temperature
        chunk_size = ctx.chunk_size
        batch_size = ctx.batch_size
        
        if batch_size <= chunk_size:
            # 小批次的backward
            logits, labels, log_probs = ctx.saved_tensors
            grad_logits = torch.zeros_like(logits)
            
            # 计算log_softmax的梯度（来自logprobs）
            grad_log_probs = torch.zeros_like(log_probs)
            grad_log_probs.scatter_add_(dim=-1, index=labels.unsqueeze(-1), src=grad_output_logprobs.unsqueeze(-1))
            
            # 计算entropy的梯度
            probs = log_probs.exp()
            grad_entropy = grad_output_entropy.unsqueeze(-1) * (-probs * log_probs - probs)
            grad_log_probs += grad_entropy
            
            # 通过log_softmax反向传播到logits
            grad_logits = grad_log_probs - probs * grad_log_probs.sum(dim=-1, keepdim=True)
            grad_logits = grad_logits / temperature
            
            return grad_logits, None, None, None
        
        # 分块处理backward
        # 创建完整的grad_logits tensor
        first_chunk_logits = ctx.chunk_info[0][2]
        grad_logits = torch.zeros_like(first_chunk_logits)
        if len(ctx.chunk_info) > 1:
            # 计算完整batch的大小
            total_size = sum(chunk_end - chunk_start for chunk_start, chunk_end, _, _, _ in ctx.chunk_info)
            grad_logits = torch.zeros(total_size, *first_chunk_logits.shape[1:], device=first_chunk_logits.device, dtype=first_chunk_logits.dtype)
        
        start_idx = 0
        for i, (chunk_start, chunk_end, chunk_logits, chunk_labels, chunk_log_probs) in enumerate(ctx.chunk_info):
            chunk_size_actual = chunk_end - chunk_start
            chunk_grad_output_logprobs = grad_output_logprobs[start_idx:start_idx + chunk_size_actual]
            chunk_grad_output_entropy = grad_output_entropy[start_idx:start_idx + chunk_size_actual]
            
            # 计算当前chunk的梯度
            grad_log_probs = torch.zeros_like(chunk_log_probs)
            grad_log_probs.scatter_add_(dim=-1, index=chunk_labels.unsqueeze(-1), src=chunk_grad_output_logprobs.unsqueeze(-1))
            
            # 计算entropy的梯度
            probs = chunk_log_probs.exp()
            grad_entropy = chunk_grad_output_entropy.unsqueeze(-1) * (-probs * chunk_log_probs - probs)
            grad_log_probs += grad_entropy
            
            # 通过log_softmax反向传播到logits
            chunk_grad_logits = grad_log_probs - probs * grad_log_probs.sum(dim=-1, keepdim=True)
            chunk_grad_logits = chunk_grad_logits / temperature
            
            # 将chunk梯度放到正确的位置
            grad_logits[chunk_start:chunk_end] = chunk_grad_logits
            start_idx += chunk_size_actual
        
        return grad_logits, None, None, None


def gather_logprobs_entropy_memory_efficient(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 1024,
):
    """
    内存高效的gather_logprobs_entropy实现，在backward阶段也进行分块处理
    
    Args:
        logits: 模型输出的logits [batch_size, seq_len, vocab_size]
        labels: 目标标签 [batch_size, seq_len]
        temperature: 温度参数
        chunk_size: 分块大小
        
    Returns:
        log_probs_labels: 对应的log概率 [batch_size, seq_len]
        entropy: 熵值 [batch_size, seq_len]
    """
    return MemoryEfficientGatherLogprobsEntropy.apply(logits, labels, temperature, chunk_size)


def gather_logprobs_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 1024,
):
    batch_size = logits.shape[0]

    if batch_size <= chunk_size:
        return _gather_logprobs_entropy(logits, labels, temperature)

    log_probs_labels_list = []
    entropy_list = []

    for i in range(0, batch_size, chunk_size):
        end_idx = min(i + chunk_size, batch_size)
        chunk_logits = logits[i:end_idx]
        chunk_labels = labels[i:end_idx]

        chunk_log_probs, chunk_entropy = _gather_logprobs_entropy(
            chunk_logits, chunk_labels, temperature
        )

        log_probs_labels_list.append(chunk_log_probs)
        entropy_list.append(chunk_entropy)

    return torch.cat(log_probs_labels_list), torch.cat(entropy_list)


@torch.no_grad()
def masked_normalization(
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dim=None,
    unbiased=False,
    eps=1e-5,
    high_precision=True,
    all_reduce=True,
    reduce_group=None,
):
    dtype = torch.float64 if high_precision else torch.float32
    x = x.to(dtype)
    if dim is None:
        dim = tuple(range(len(x.shape)))
    if mask is None:
        factor = torch.tensor(
            np.prod([x.shape[d] for d in dim]), dtype=dtype, device=x.device
        )
    else:
        mask = mask.to(dtype)
        x = x * mask
        factor = mask.sum(dim, keepdim=True)
    x_sum = x.sum(dim=dim, keepdim=True)
    x_sum_sq = x.square().sum(dim=dim, keepdim=True)
    if dist.is_initialized() and all_reduce:
        dist.all_reduce(factor, op=dist.ReduceOp.SUM, group=reduce_group)
        dist.all_reduce(x_sum, op=dist.ReduceOp.SUM, group=reduce_group)
        dist.all_reduce(
            x_sum_sq,
            op=dist.ReduceOp.SUM,
            group=reduce_group,
        )
    mean = x_sum / factor
    meansq = x_sum_sq / factor
    var = meansq - mean**2
    if unbiased:
        var *= factor / (factor - 1)
    return ((x - mean) / (var.sqrt() + eps)).float()


def ppo_actor_loss_fn(
    logprobs: torch.Tensor,
    proximal_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    loss_mask: torch.Tensor,
    c_clip: Optional[float] = None,
    behav_imp_weight_cap: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    When decoupled loss is disabled:
    1. if recompute logp, both old_logprobs and proximal_logprobs are recomputed logp;
    2. if no recomputation, both old_logp and proximal_logprobs are produced by the inference backend.

    When decoupled loss is enabled, proximal_logprobs is the recomputed logp,
    old_logprobs is produced by the inference engine.
    """
    loss_mask_count = loss_mask.count_nonzero() or 1
    ratio = torch.where(loss_mask, torch.exp(logprobs - proximal_logprobs), 0)
    clipped_ratio = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    clip_mask = pg_loss1.detach() < pg_loss2.detach()
    pg_loss = torch.max(pg_loss1, pg_loss2)
    if c_clip is not None:
        assert c_clip > 1.0, c_clip
        pg_loss3 = torch.sign(advantages) * c_clip * advantages
        dual_clip_mask = pg_loss3.detach() < pg_loss.detach()
        pg_loss = torch.min(pg_loss, pg_loss3)
    else:
        dual_clip_mask = torch.zeros_like(clip_mask)
    behav_kl = proximal_logprobs - old_logprobs
    behav_imp_weight = behav_kl.exp()
    behav_mask = (
        (behav_imp_weight <= behav_imp_weight_cap).logical_and(loss_mask)
        if behav_imp_weight_cap is not None
        else loss_mask
    )
    behav_kl = torch.where(behav_mask, behav_kl, 0.0)
    behav_imp_weight = torch.where(behav_mask, behav_imp_weight, 0.0)
    pg_loss = pg_loss * behav_imp_weight
    logging_loss = pg_loss.detach()
    pg_loss = torch.where(loss_mask, pg_loss, 0).sum() / loss_mask_count
    clip_mask.logical_and_(loss_mask)
    dual_clip_mask.logical_and_(loss_mask)
    stat = dict(
        loss=logging_loss,
        importance_weight=ratio.detach(),
        approx_kl=(logprobs - proximal_logprobs).detach(),
        clip_mask=clip_mask,
        dual_clip_mask=dual_clip_mask,
    )
    if proximal_logprobs is not None:
        stat["behave_imp_weight"] = behav_imp_weight
        stat["behave_approx_kl"] = behav_kl
        stat["behave_mask"] = behav_mask
    return pg_loss, stat
