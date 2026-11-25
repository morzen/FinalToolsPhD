import time
import psutil
import GPUtil
import subprocess
import sys
import os

class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        
    def start(self):
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
    def stop(self, path):
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024
        
        print("\n" + "="*50)
        print("RESOURCE USAGE REPORT" + str(path))
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

# scriptpath = "../1.XGboost/False_flag.py"

# scriptpath = "../2.GNN/False_flag.py"

# scriptpath = "../3.SiameseNetwork/False_flag.py"

# scriptpath = "../4.Attention-Based_MIL/eval.py"

# scriptpath = "../5.RF_FeatureEngineering/False_flag.py"

# scriptpath = "../6.hybrid/hybrid_eval.py"

# scriptpath = "../featureMatching/false_flag.py"
# scriptpath = "../Jaccards_Correlation/false_flag.py"
# scriptpath = "../TVD_Total_Variation_Distance/false_flag.py"


# Usage with subprocess
monitor = ResourceMonitor()
monitor.start()

# Run the script and let its output print directly to stdout
try:
    # Use stdout=None (default) to let subprocess output directly to console
    # This way the script's output appears immediately and in order
    result = subprocess.run([sys.executable, scriptpath])
    
    # If you need to capture and ensure ordering, you can use this approach:
    # result = subprocess.run([sys.executable, scriptpath], 
    #                        capture_output=True, 
    #                        text=True)
    # print(result.stdout, end='')  # Print script output first
    # if result.stderr:
    #     print(result.stderr, end='')
        
except Exception as e:
    print(f"Error running script: {e}")

monitor.stop(scriptpath)