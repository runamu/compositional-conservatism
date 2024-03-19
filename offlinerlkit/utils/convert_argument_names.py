import os
import re
import ipdb

def convert_python_files(directory):
    # Find all python files in the directory
    # use tqdm
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                # Open the python file and read its contents
                with open(file_path, "r") as f:
                    contents = f.read()
                if "parser.add_argument" not in contents:
                    continue

                print(f"Converting {file_path}")
                # Replace all instances of "-" with "_" in lines matching the format parser.add_argument("--some-argument-name", ...)
                contents = re.sub(r'([#\s]*parser\.add_argument\(\s*")(--)([\w+-?]*\w+",)',
                                lambda m: f'{m.group(1)}{m.group(2)}{m.group(3).replace("-", "_")}',
                                contents)
                # Write the modified contents back to the file
                with open(file_path, "w") as f:
                    f.write(contents)
    print("Done!")

# Call the function with the directory you want to modify
convert_python_files("run_example")
convert_python_files("tune_example")
convert_python_files("eval_example")
