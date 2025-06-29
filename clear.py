import gc
import torch

def clear_cuda_cache():
    # 清除 Python 层面的无用变量
    gc.collect()

    # 如果有 CUDA 支持
    if torch.cuda.is_available():
        torch.cuda.empty_cache()       # 清空未使用的缓存块
        torch.cuda.ipc_collect()       # 清理 inter-process communication 缓存
        torch.cuda.reset_peak_memory_stats()  # 可选：重置峰值显存记录
        torch.cuda.synchronize()       # 同步确保释放生效
        print(f"[INFO] Cleared CUDA memory. Current usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    else:
        print("[INFO] CUDA not available.")

# 示例用法
if __name__ == "__main__":
    import torch

    # 当前被PyTorch张量占用的显存（单位字节）
    allocated = torch.cuda.memory_allocated()
    print(f"Allocated memory: {allocated / 1024**2:.2f} MB")
    
    # 当前PyTorch缓存池占用的显存（单位字节）
    cached = torch.cuda.memory_reserved()  # 在新版 PyTorch 中改名为 memory_reserved()
    print(f"Cached memory: {cached / 1024**2:.2f} MB")
    
    # 显卡总显存（单位字节）
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"Total GPU memory: {total / 1024**2:.2f} MB")
    
    # 当前设备显存的峰值已分配内存
    peak_allocated = torch.cuda.max_memory_allocated()
    print(f"Peak allocated memory: {peak_allocated / 1024**2:.2f} MB")
    
    # 当前设备显存的峰值缓存内存
    peak_cached = torch.cuda.max_memory_reserved()
    print(f"Peak cached memory: {peak_cached / 1024**2:.2f} MB")

    clear_cuda_cache()
