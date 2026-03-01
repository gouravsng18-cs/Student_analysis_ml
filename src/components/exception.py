import sys
from logger import logging
def error_messsage_detail(error,error_detail:sys): # type: ignore
    _,_,exe_tb = error_detail.exc_info()
    file_name = exe_tb.tb_frame.f_code.co_filename # type: ignore
    error_message = "An error has occured in [{0}] line number [{1}] error message[{2}]".format(
     file_name,exe_tb.tb_lineno,str(error)) # type: ignore
    return error_message


class custom_exception(Exception):
    def __init__(self, error_message,error_detail:sys): # type: ignore
        super().__init__(error_message)
        self.error_message = error_messsage_detail(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message
    
if __name__ == "__main__":
    logging.info("Logging has started")
    