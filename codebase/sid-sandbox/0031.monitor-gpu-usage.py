import tkinter as tk
import threading
import time
import cupy as cp

class GPUMonitorApp:
    usage_label = None  # Class attribute to hold the label widget
    gpu_memory_pool = None  # Class attribute to hold the GPU memory pool
    root = None  # Class attribute to hold the Tkinter root window

    @classmethod
    def create_gui(cls):
        cls.root = tk.Tk()
        cls.root.title("GPU Monitor")

        cls.gpu_memory_pool = cp.get_default_memory_pool()

        label = tk.Label(cls.root, text="Current GPU Usage:")
        label.pack(pady=10)

        cls.usage_label = tk.Label(cls.root, text="0 MB")
        cls.usage_label.pack()

        monitor_thread = threading.Thread(target=cls.update_gpu_usage_thread, daemon=True)
        monitor_thread.start()

        cls.root.mainloop()

    @classmethod
    def update_gpu_usage_thread(cls):
        while True:
            total_mempool = cls.gpu_memory_pool.total_bytes() / (1024 * 1024)
            used_mempool = cls.gpu_memory_pool.used_bytes() / (1024 * 1024)
            used_gpu_mb = used_mempool / (1024 * 1024)  # Convert to MB

            cls.update_gpu_label(f"{used_gpu_mb:.2f} MB")
            time.sleep(1)  # Update GPU usage every second

    @classmethod
    def update_gpu_label(cls, text):
        cls.usage_label.config(text=text)
        cls.usage_label.master.title(f"GPU Monitor - {text}")

def add():
    c = 1 + 2
    print("add() function called")

def main():
    GPUMonitorApp.create_gui()

if __name__ == "__main__":
    app_thread = threading.Thread(target=main, daemon=True)
    app_thread.start()

    # Wait for a moment to ensure the GUI thread is running
    time.sleep(1)

    # Main program can now call add() or perform other tasks concurrently
    add()

    # Wait for the GUI thread to finish
    app_thread.join()
