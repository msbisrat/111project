# This code is about creating a custom way to handle and display errors 
# in a more informative and structured manner—especially useful 
# in machine learning or large-scale projects.



import sys
import os
# sys: This is a Python module used to get detailed error information.

# logging: You're importing the logger you set up earlier 
# (to write errors into a log file).

def error_message_detail(error , error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python scripts name [{0}] line number [{1}] error message[{2}]".format(
     file_name , exc_tb.tb_lineno , str(error))
    
    return error_message
# error: The error that occurred
# error_detail: Info from the sys module to find out more
# (like which line it happened on)    
# exc_info() gets traceback info: what happened, where it happened, etc.
# _ are just placeholders (we only care about exc_tb = traceback)
#    file_name = exc_tb.tb_frame.f_code.co_filename
# This finds the file name where the error occurred.


class CustomException(Exception):
    def __init__(self , error_message , error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message , error_details)
    
    def __str__(self)    :
        return self.error_message
    
    