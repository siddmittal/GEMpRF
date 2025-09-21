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
Logger.print_yellow_message(base64.b64decode(b'Cj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT'
b'09PT09PT09PT09PT09PT09PQpHRU0gcFJGIEFuYWx5c2lzIC0gVW5yZWx'
b'lYXNlZCBWZXJzaW9uIGZvciBUZXN0aW5nIE9ubHkKPT09PT09PT09PT09PT09'
b'PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Cgp'
b'AQXV0aG9yICAgOiBTaWRkaGFydGggTWl0dGFsCkBWZXJzaW9uICA6IDEuMApAQ2'
b'9udGFjdCAgOiBzaWRkaGFydGgubWl0dGFsQG1lZHVuaXdpZW4uYWMuYXQKQExpY'
b'2Vuc2UgIDogKEMpIENvcHlyaWdodCAyMDI0LTIwMjUsIE1lZGljYWwgVW5pdmVyc'
b'2l0eSBvZiBWaWVubmEKCldBUk5JTkc6ClRoaXMgaXMgYW4gdW5yZWxlYXNlZCB2Z'
b'XJzaW9uIGZvciB0ZXN0aW5nIHB1cnBvc2VzLgpJdHMgZGlzdHJpYnV0aW9uIGlzI'
b'HN0cmljdGx5IHByb2hpYml0ZWQuCgo9PT09PT09PT09PT09PT09PT09PT09PT09PT0'
b'9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0K').decode('utf-8'), print_file_name=False)

