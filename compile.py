import os
import torch
import torch._dynamo as dynamo
# from torch._dynamo.utils import CompileCounter
torch._inductor.config.debug = True
torch._dynamo.config.verbose = True
logits = torch.randn(32, 1000, requires_grad=True).cuda()  # batch_size=32, vocab_size=1000
labels = torch.randint(0, 1000, (32,)).cuda()


# 方法6: 保存中间表示到文件

def debug_backend(gm, example_inputs):
    print("FX图代码:")
    print(gm.code)
    print("图结构:")
    gm.graph.print_tabular()
    return gm

@torch.compile(backend="inductor")
def _gather_logprobs_save_debug(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0
):
    log_probs = torch.nn.functional.log_softmax(logits.float() / temperature, dim=-1).cuda()
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1).cuda()
    return log_probs_labels

# print("\n=== 保存调试信息到文件 ===")
result3 = _gather_logprobs_save_debug(logits, labels)

loss = -result3.mean()
loss.backward()

# print("调试信息已保存到 ./torch_compile_debug 目录")

