import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import shutil

temp_dir = './temp_directory'

def replace_in_file(file_path, string1, string2):
    # Open the file and read all lines
    with open(file_path, 'r') as file:
        content = file.read()

    if string1:  # If string1 is not empty, replace all occurrences
        # Replace all occurrences of string1 with string2
        modified_content = content.replace(string1, string2)
    else:  # If string1 is empty, prepend string2 to the start of the file
        # Prepend string2 to the beginning of the content
        modified_content = string2 + content

    # Open the file again and overwrite it with the modified content
    with open(file_path, 'w') as file:
        file.write(modified_content)

# Ensure the temporary directory exists
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Initialize an empty list to store tuples
input_files = []

# Open the file in read mode
with open('inputs.txt', 'r') as file:
    # Read each line in the file
    for line in file:
        # Strip the line of leading/trailing spaces and split it
        parts = line.strip().split()
        
        # Ensure there is at least one part, and assign "" if there is no second part
        if len(parts) == 1:
            input_files.append((parts[0], ""))  # Only one part, second is an empty string
        elif len(parts) == 2:
            input_files.append((parts[0], parts[1]))  # Two parts, normal tuple

# Print the list of tuples
print(input_files)
output_dir = "output"  # Directory to save output
pumpkin_command = "MZN_SOLVER_PATH=/Users/davidr86/Documents/Pumpkin/minizinc /Applications/MiniZincIDE.app/Contents/Resources/minizinc --solver /Users/davidr86/Documents/Pumpkin/minizinc/pumpkin.msc --output-time --statistics"

commands = [
    ("decomp",f"{pumpkin_command}", []),
    ("regin", f"{pumpkin_command}", [("global_cardinality_low_up(", "pumpkin_gcc_regin("), ("", 'include "pumpkin_gcc.mzn";\n')]),
    ("basic_filter", f"{pumpkin_command}", [("global_cardinality_low_up(", "pumpkin_gcc_basic_filter("), ("", 'include "pumpkin_gcc.mzn";\n')])
]  # List of command templates

print(commands)

commands_to_run = []

# Iterate over each command (or each element in input_files)
for com in commands:
    for ifile in input_files:

        input_file_path = ifile[0]
    
        path, ext = os.path.splitext(os.path.basename(input_file_path))

        # Define the destination path in the temporary directory (using the file name from input_file)
        destination_path = os.path.join(temp_dir, path+f"_{com[0]}"+ext)
    
        # Copy the file to the temporary directory, overwriting if the file exists
        shutil.copy(input_file_path, destination_path)

        output_path = f"{os.path.basename(destination_path)}-{os.path.basename(ifile[1])}.txt"
        command = f"{com[1]} {destination_path} {ifile[1]}"

        commands_to_run.append(("com[0]", command, output_path))

        print(f"Copied {input_file_path} to {destination_path}")

        for replace in com[2]:
            replace_in_file(destination_path, replace[0], replace[1])


print(f"ctr: {commands_to_run}")


num_cores = 6  # Number of cores to use


def run_command(name, command, output_file):
    """Runs a command and saves the output to a file."""
    try:
        with open(output_file, "w") as f:
            subprocess.run(command, shell=True, stdout=f, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {command}\nError: {e}")

def main():
    # Generate all combinations of input files, data files, and commands

    # Use ThreadPoolExecutor to run commands on multiple cores
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(run_command, name, cmd, outfile) for name, cmd, outfile in commands_to_run]

        # Wait for all futures to complete
        for future in futures:
            future.result()

if __name__ == "__main__":
    main()
