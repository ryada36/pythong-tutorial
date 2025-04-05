# File handling
import io

# without "with"
# file = open("./README.md", 'a+')
# try:
#     content = file.read()
#     print(content)
#     file.write("Written from code \n")
#     file.seek(0)  # Move cursor back to the beginning again
#     content = file.read()
#     print(content)
# finally:
#     file.close()


# # writing with "with" keyword 
# with open("./README.md","a+") as file:
#     print(file.read())

# read from one file and write to other
# with open("./README.md", "r") as source_file, open("./README_copy.md", "w+") as dest_file:
#     content = source_file.read()
#     dest_file.write(content)

# reading a file line by line in chunks/streams
# with open("./README.md", "r") as file, open("./README_copy.md", "w+") as dest_file:
#     for chunk in iter(lambda: file.read(4096), ''):
#         dest_file.write(chunk)
#         print(chunk)

# reading writing with buffered IO
# with open("./README.md", "rb") as file, open("./README_copy.md", "wb") as dest_file:
#     with io.BufferedReader(file) as reader, io.BufferedWriter(dest_file) as writer:
#         for chunk in iter(lambda: reader.read(4096), b""):
#             writer.write(chunk)
    
