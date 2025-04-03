# Custom exception class
class CustomError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

   
try:
    print("Doing error handling")
    raise CustomError("This is a custom error")
except CustomError as e:
    print(f"Error occurred: {e}")
finally:
    print("Finally block executed")