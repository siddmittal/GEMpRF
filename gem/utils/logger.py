# -*- coding: utf-8 -*-

"""

"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2024-2025, Siddharth Mittal",
"@Desc    :   None",
        
"""
import inspect
import base64
class Logger:
    @classmethod
    def print_red_message(cls, message, print_file_name=True):        
        print(f"\033[91m{message}\033[0m")
        if (print_file_name):
            frame = inspect.currentframe().f_back
            caller_filename = inspect.getframeinfo(frame).filename
            print(f"\033[91mFile: {caller_filename}\033[0m")
    
    @classmethod
    def print_orange_message(cls, message, print_file_name=True):
        print(f"\033[38;5;208m{message}\033[0m")
        if (print_file_name):
            frame = inspect.currentframe().f_back
            caller_filename = inspect.getframeinfo(frame).filename   
            print(f"\033[38;5;208mFile: {caller_filename}\033[0m")

    @classmethod
    def print_blue_message(cls, message, print_file_name=True):             
        print(f"\033[94m{message}\033[0m")
        if (print_file_name):
            frame = inspect.currentframe().f_back
            caller_filename = inspect.getframeinfo(frame).filename   
            print(f"\033[94mFile: {caller_filename}\033[0m")            

    @classmethod
    def print_green_message(cls, message, print_file_name=True):             
        print(f"\033[92m{message}\033[0m")
        if (print_file_name):
            frame = inspect.currentframe().f_back
            caller_filename = inspect.getframeinfo(frame).filename   
            print(f"\033[92mFile: {caller_filename}\033[0m")

    @classmethod
    def print_yellow_message(cls, message, print_file_name=True):             
        print(f"\033[93m{message}\033[0m")
        if print_file_name:
            frame = inspect.currentframe().f_back
            caller_filename = inspect.getframeinfo(frame).filename   
            print(f"\033[93mFile: {caller_filename}\033[0m")

# # base64.b64encode(original_str.encode('utf-8'))
val = (b'PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09CkdFTSBwUkYgQW5hbHlzaXMgU29mdHdhcmUKPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09CgpAQXV0aG9yICAgOiBTaWRkaGFydGggTWl0dGFsCkBDb250YWN0ICA6IHNpZGRoYXJ0aC5taXR0YWxAbWVkdW5pd2llbi5hYy5hdApATGljZW5zZSAgOiAoQykgQ29weXJpZ2h0IDIwMjMtMjAyNSwgTWVkaWNhbCBVbml2ZXJzaXR5IG9mIFZpZW5uYSAKQENpdGUgRE9JIDogaHR0cHM6Ly9kb2kub3JnLzEwLjEwMTYvai5tZWRpYS4yMDI1LjEwMzg5MQoKUGFwZXIgVGl0bGU6IEdFTS1wUkY6IEdQVS1FbXBvd2VyZWQgTWFwcGluZyBvZiBQb3B1bGF0aW9uIApSZWNlcHRpdmUgRmllbGRzIGZvciBMYXJnZS1TY2FsZSBmTVJJIEFuYWx5c2lzCj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PQ==')
Logger.print_yellow_message(base64.b64decode(val).decode('utf-8'), print_file_name=False)
