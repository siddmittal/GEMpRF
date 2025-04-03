# -*- coding: utf-8 -*-

"""

"@Author  :   Siddharth Mittal",
"@Version :   1.0",
"@Contact :   siddharth.mittal@meduniwien.ac.at",
"@License :   (C)Copyright 2024, Siddharth Mittal",
"@Desc    :   None",
        
"""
import inspect


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
