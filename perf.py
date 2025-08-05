import torch
import contextlib
import os

class NVTXProfiler:
    def __init__(self, enabled=True, target_step=None):
        self.enabled = enabled and torch.cuda.is_available()
        self.target_step = target_step
        self.current_step = 0

    @contextlib.contextmanager
    def range(self, name, step=None):
        """NVTX范围标记"""
        if self.enabled and (self.target_step is None or step == self.target_step):
            torch.cuda.nvtx.range_push(name)
            try:
                yield
            finally:
                torch.cuda.nvtx.range_pop()
        else:
            yield

    def mark(self, name, step=None):
        """NVTX点标记"""
        if self.enabled and (self.target_step is None or step == self.target_step):
            torch.cuda.nvtx.mark(name)

#   profiler = NVTXProfiler(target_step=5)  # 针对第5个step

#   with profiler.range(f"train_step_complete_step_{self.current_step}", self.current_step):
#       if request.handle_name == "train_step":
#           with profiler.range(f"train_step_interface_call_step_{self.current_step}", self.current_step):
#               res = self._interface.train_step(
#                   self._model,
#                   data,
#                   mb_spec=rpc.mb_spec,
#               )


# export NVTX_PYTHON_ENABLED=1

# # 使用nsys进行性能分析，仅捕获GPU kernels和NVTX标记
# nsys profile \
#     --trace=cuda,nvtx \
#     --duration=60 \
#     --delay=30 \
#     --capture-range=nvtx \
#     --nvtx-capture="train_step_complete_step_5" \
#     --output=areal_step5_profile \
#     --export=sqlite \
#     --force-overwrite=true \
#     python train_script.py

# # 生成轻量级报告
# nsys stats --report=gputrace,osrtrace,nvtxsum areal_step5_profile.nsys-rep > step5_summary.txt
