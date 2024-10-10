import psutil
import tkinter as tk
import threading
import time

class RAMMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RAM Monitor")

        self.label = tk.Label(root, text="Current RAM Usage:")
        self.label.pack(pady=10)

        self.usage_label = tk.Label(root, text="0 MB")
        self.usage_label.pack()

        self.increase_button = tk.Button(root, text="Increase RAM Usage", command=self.increase_ram_usage)
        self.increase_button.pack(pady=10)

        self.decrease_button = tk.Button(root, text="Decrease RAM Usage", command=self.decrease_ram_usage)
        self.decrease_button.pack()

        self.current_usage = 0  # Initial RAM usage in MB

        # Create a thread to continuously monitor RAM usage
        self.monitor_thread = threading.Thread(target=self.update_ram_usage_thread, daemon=True)
        self.monitor_thread.start()

    def increase_ram_usage(self):
        # Simulate increasing RAM usage by allocating memory
        self.current_usage += 10  # Increase RAM usage by 10 MB
        self.update_ram_usage()

    def decrease_ram_usage(self):
        # Simulate decreasing RAM usage by deallocating memory
        self.current_usage -= 10  # Decrease RAM usage by 10 MB
        if self.current_usage < 0:
            self.current_usage = 0
        self.update_ram_usage()

    def update_ram_usage_thread(self):
        while True:
            # Get and display current RAM usage in a thread-safe manner
            ram_info = psutil.virtual_memory()
            used_ram_mb = ram_info.used / (1024 * 1024)  # Convert to MB
            self.root.after(0, self.update_ram_label, f"{used_ram_mb:.2f} MB")
            time.sleep(1)  # Update RAM usage every second

    def update_ram_label(self, text):
        self.usage_label.config(text=text)
        self.root.title(f"RAM Monitor - {text}")

def add():
    c = 1 + 2
    print("add() function called")

def main():
    root = tk.Tk()
    app = RAMMonitorApp(root)
    
    # Create a button to call the add() function
    add_button = tk.Button(root, text="Call add() function", command=add)
    add_button.pack()

    root.mainloop()

if __name__ == "__main__":
    main()
