import os

def get_directory_names(path):
    # List all directories in the given path
    directories = [d for d in os.listdir(path) 
                   if os.path.isdir(os.path.join(path, d)) and d != '__pycache__' and d != ".ipynb_checkpoints" and d != "AA12345"]
    return directories