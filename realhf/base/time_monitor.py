# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0 (the "License").

import json
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from realhf.base import name_resolve, names


@dataclass
class WorkerTimeEvent:
    """Worker时间事件记录"""
    worker_name: str
    worker_type: str
    worker_index: int
    event_type: str  # "start", "pause", "resume", "complete", "error", "poll_start", "poll_end"
    timestamp: float  # 时间戳
    status: str  # worker状态
    duration: Optional[float] = None  # 事件持续时间（如果适用）
    metadata: Optional[Dict] = None  # 额外的元数据


class WorkerTimeMonitor:
    """Worker时间监控器"""
    
    def __init__(self, experiment_name: str, trial_name: str, worker_name: str, worker_type: str, worker_index: int):
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.worker_name = worker_name
        self.worker_type = worker_type
        self.worker_index = worker_index
        
        # 记录事件历史
        self.events: List[WorkerTimeEvent] = []
        self.start_time = time.time()
        self.last_poll_start = None
        
    def record_event(self, event_type: str, status: str, duration: Optional[float] = None, metadata: Optional[Dict] = None):
        """记录一个时间事件"""
        timestamp = time.time()
        event = WorkerTimeEvent(
            worker_name=self.worker_name,
            worker_type=self.worker_type,
            worker_index=self.worker_index,
            event_type=event_type,
            timestamp=timestamp,
            status=status,
            duration=duration,
            metadata=metadata or {}
        )
        
        self.events.append(event)
        
        # 通过name_resolve发布事件
        self._publish_event(event)
        
    def record_poll_start(self):
        """记录poll开始"""
        self.last_poll_start = time.time()
        self.record_event("poll_start", "RUNNING")
        
    def record_poll_end(self, sample_count: int = 0, batch_count: int = 0):
        """记录poll结束"""
        if self.last_poll_start is not None:
            duration = time.time() - self.last_poll_start
            metadata = {
                "sample_count": sample_count,
                "batch_count": batch_count,
                "poll_duration": duration
            }
            self.record_event("poll_end", "RUNNING", duration=duration, metadata=metadata)
            self.last_poll_start = None
            
    def record_status_change(self, old_status: str, new_status: str):
        """记录状态变更"""
        event_type_map = {
            "RUNNING": "start",
            "PAUSED": "pause", 
            "COMPLETED": "complete",
            "ERROR": "error",
            "INTERRUPTED": "interrupt"
        }
        
        event_type = event_type_map.get(new_status, "status_change")
        metadata = {"old_status": old_status, "new_status": new_status}
        self.record_event(event_type, new_status, metadata=metadata)
        
    def _publish_event(self, event: WorkerTimeEvent):
        """通过name_resolve发布事件"""
        try:
            key = names.worker_key(
                experiment_name=self.experiment_name,
                trial_name=self.trial_name,
                key=f"time_monitor/{self.worker_name}/latest"
            )
            
            # 发布最新事件
            name_resolve.add(
                key,
                value=json.dumps(asdict(event)),
                replace=True,
                delete_on_exit=False
            )
            
            # 同时发布到历史记录
            history_key = names.worker_key(
                experiment_name=self.experiment_name,
                trial_name=self.trial_name,
                key=f"time_monitor/{self.worker_name}/history/{event.timestamp}"
            )
            
            name_resolve.add(
                history_key,
                value=json.dumps(asdict(event)),
                replace=True,
                delete_on_exit=False
            )
            
        except Exception as e:
            # 静默失败，不影响worker正常运行
            pass
            
    def get_runtime_stats(self) -> Dict:
        """获取运行时统计信息"""
        current_time = time.time()
        total_runtime = current_time - self.start_time
        
        # 计算各状态的时间
        status_times = {}
        last_event_time = self.start_time
        current_status = "READY"
        
        for event in self.events:
            if event.event_type in ["start", "pause", "complete", "error", "interrupt"]:
                # 记录上一个状态的持续时间
                if current_status not in status_times:
                    status_times[current_status] = 0
                status_times[current_status] += event.timestamp - last_event_time
                
                # 更新当前状态
                current_status = event.status
                last_event_time = event.timestamp
        
        # 添加当前状态到现在的时间
        if current_status not in status_times:
            status_times[current_status] = 0
        status_times[current_status] += current_time - last_event_time
        
        return {
            "total_runtime": total_runtime,
            "status_times": status_times,
            "event_count": len(self.events),
            "current_status": current_status
        } 