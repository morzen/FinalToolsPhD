import time
import psutil
import GPUtil

class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        
    def start(self):
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
    def stop(self):
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024
        
        print("\n" + "="*50)
        print("RESOURCE USAGE REPORT")
        print("="*50)
        
        # Time
        print(f"Execution Time: {end_time - self.start_time:.2f} seconds")
        
        # Memory
        print(f"Memory Usage: {end_memory - self.start_memory:.2f} MB")
        print(f"Current Memory: {end_memory:.2f} MB")
        
        # CPU
        cpu_percent = self.process.cpu_percent(interval=0.1)
        print(f"CPU Usage: {cpu_percent}%")
        
        # System-wide CPU
        sys_cpu = psutil.cpu_percent(interval=0.1)
        print(f"System CPU: {sys_cpu}%")
        
        # GPU (if available)
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                print(f"\nGPU {gpu.id} ({gpu.name}):")
                print(f"  Utilization: {gpu.load*100:.1f}%")
                print(f"  Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
        except:
            print("\nGPU: Not available or not detected")

# Usage
monitor = ResourceMonitor()
monitor.start()

#the code


monitor.stop()