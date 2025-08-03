import torch
import torch.nn.functional as F
from functional import gather_logprobs, gather_logprobs_memory_efficient


def test_gather_logprobs_correctness():
    """测试内存高效版本的gather_logprobs与原始版本的一致性"""
    
    # 测试参数
    batch_sizes = [256]#, 1024, 2048, 4096]
    seq_len = 128
    vocab_size = 320
    temperatures = [0.5, 1.0, 2.0]
    chunk_sizes = [256, 512]#, 1024]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("开始测试gather_logprobs内存高效版本的正确性...")
    
    for batch_size in batch_sizes:
        for temperature in temperatures:
            for chunk_size in chunk_sizes:
                if batch_size < chunk_size:
                    continue
                    
                print(f"测试 batch_size={batch_size}, temperature={temperature}, chunk_size={chunk_size}")
                
                # 生成测试数据
                torch.manual_seed(42)
                logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
                labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
                
                # 原始版本
                logits_orig = logits.clone().detach().requires_grad_(True)
                result_orig = gather_logprobs(logits_orig, labels, temperature, chunk_size)
                
                # 内存高效版本
                logits_mem = logits.clone().detach().requires_grad_(True)
                result_mem = gather_logprobs_memory_efficient(logits_mem, labels, temperature, chunk_size)
                
                # 检查forward结果一致性
                forward_diff = torch.abs(result_orig - result_mem).max().item()
                print(f"  Forward差异: {forward_diff:.2e}")
                assert forward_diff < 1e-5, f"Forward结果不一致: {forward_diff}"
                
                # 检查backward梯度一致性
                # 创建相同的梯度输出
                grad_output = torch.randn_like(result_orig)
                
                # 原始版本backward
                result_orig.backward(grad_output)
                grad_orig = logits_orig.grad.clone()
                
                # 内存高效版本backward
                result_mem.backward(grad_output)
                grad_mem = logits_mem.grad.clone()
                
                # 检查梯度一致性
                grad_diff = torch.abs(grad_orig - grad_mem).max().item()
                print(f"  Backward差异: {grad_diff:.2e}")
                assert grad_diff < 1e-5, f"Backward梯度不一致: {grad_diff}"
                
                print(f"  ✓ 测试通过")
    
    print("所有测试通过！")


def test_memory_usage():
    """测试内存使用情况"""
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过内存测试")
        return
        
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    
    # 大batch测试
    batch_size = 32
    seq_len = 256
    vocab_size = 3200
    temperature = 1.0
    chunk_size = 1024
    
    print(f"测试内存使用: batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}")
    
    # 生成测试数据
    torch.manual_seed(42)
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # 测试原始版本内存使用
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    result_orig = gather_logprobs(logits, labels, temperature, chunk_size)
    grad_output = torch.randn_like(result_orig)
    result_orig.backward(grad_output)
    
    memory_orig = torch.cuda.max_memory_allocated() / 1024**3  # GB
    print(f"原始版本峰值内存: {memory_orig:.2f} GB")
    
    # 测试内存高效版本内存使用
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    logits_mem = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
    labels_mem = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    result_mem = gather_logprobs_memory_efficient(logits_mem, labels_mem, temperature, chunk_size)
    grad_output_mem = torch.randn_like(result_mem)
    result_mem.backward(grad_output_mem)
    
    memory_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
    print(f"内存高效版本峰值内存: {memory_mem:.2f} GB")
    print(f"内存节省: {((memory_orig - memory_mem) / memory_orig * 100):.1f}%")


def test_edge_cases():
    """测试边界情况"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("测试边界情况...")
    
    # 测试小batch
    batch_size = 64
    seq_len = 32
    vocab_size = 1000
    temperature = 1.0
    chunk_size = 1024
    
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    result_orig = gather_logprobs(logits, labels, temperature, chunk_size)
    result_mem = gather_logprobs_memory_efficient(logits, labels, temperature, chunk_size)
    
    assert torch.allclose(result_orig, result_mem, atol=1e-6), "小batch测试失败"
    print("  ✓ 小batch测试通过")
    
    # 测试单样本
    batch_size = 1
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    result_orig = gather_logprobs(logits, labels, temperature, chunk_size)
    result_mem = gather_logprobs_memory_efficient(logits, labels, temperature, chunk_size)
    
    assert torch.allclose(result_orig, result_mem, atol=1e-6), "单样本测试失败"
    print("  ✓ 单样本测试通过")
    
    # 测试不同温度
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    batch_size = 256
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    for temp in temperatures:
        result_orig = gather_logprobs(logits, labels, temp, chunk_size)
        result_mem = gather_logprobs_memory_efficient(logits, labels, temp, chunk_size)
        assert torch.allclose(result_orig, result_mem, atol=1e-6), f"温度{temp}测试失败"
    
    print("  ✓ 不同温度测试通过")


if __name__ == "__main__":
    print("开始测试内存高效的gather_logprobs实现...")
    
    # 运行所有测试
    test_gather_logprobs_correctness()
    # test_memory_usage()
    test_edge_cases()
    
    print("所有测试完成！") 