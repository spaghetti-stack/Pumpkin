import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import shutil
import time
import shlex
import argparse
import random


temp_dir = './temp_directory'

timeout_secs = 10 * 60 ## 20 minutes
NUM_CORES = 6-1;

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


def run_command(name, command, output_file, timeout=timeout_secs+60):
    """
    Runs a command and saves the output to a file.
    If the command takes more than `timeout` seconds, it is terminated.
    """
    print(f"Running command: {command}")
    try:
        command_parts = shlex.split(command)
        env = os.environ.copy()  # Copy the current environment
        if "=" in command_parts[0]:  # Check if the first part is an ENV=value pair
            key, value = command_parts.pop(0).split("=", 1)  # Extract and remove it
            env[key] = value
        
        with open(output_file, "w") as f:
            process = subprocess.Popen(command_parts, env=env, stderr=f, stdout=f)
            
            try:
                # Removed timeout for now. Instead it is handles by Minizinc.
                #process.wait(timeout=timeout)  # Wait for process with a timeout
                process.wait()
            except subprocess.TimeoutExpired:
                print(f"Command '{command}' timed out after {timeout} seconds.")
                process.terminate()  # Gracefully terminate the process
                try:
                    process.wait(timeout=5)  # Allow some time for cleanup
                except subprocess.TimeoutExpired:
                    process.kill()  # Force kill if it doesn't exit in time
                raise RuntimeError(f"Command '{command}' timed out and was terminated.")
    
    except Exception as e:
        print(f"Command '{command}' failed with error: {e}")


def print_status(futures, start_time, total_tasks, commands_to_run):
    # Count how many tasks have completed
    completed_tasks = sum(1 for future in futures if future.done())
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds, {completed_tasks}/{total_tasks} tasks finished")

    # Print currently running tasks
    running_tasks = [commands_to_run[i][2] for i, future in enumerate(futures) if future.running()]
    if running_tasks:
        print("Currently running tasks:")
        for task in running_tasks:
            print(f"  {task}")


def main():

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process input and output directory paths.")

    # Add arguments for input and output directories
    parser.add_argument(
        "--inputs",
        type=str,
        required=True,
        help="Path to the input files or directory."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Validate input path
    if not os.path.exists(args.inputs):
        print(f"Error: The input path '{args.inputs}' does not exist.")
        return

    output_dir = args.output_dir
    inputs = args.inputs

    # Validate or adjust output directory path
    while os.path.exists(output_dir):
        output_dir = output_dir.rstrip(os.sep) + "-new" # Remove trailing slash if it exists
        print(f"Output directory exists. Using '{output_dir}' instead.")

    try:
        os.makedirs(output_dir)
        print(f"Output directory '{output_dir}' is ready.")
    except OSError as e:
        print(f"Error: Could not create output directory '{output_dir}'. {e}")
        return

    # Confirm paths
    print(f"Inputs path: {inputs}")
    print(f"Output directory: {output_dir}")


    # Ensure the temporary directory exists
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)


    # Initialize an empty list to store tuples
    input_files = []

    # Open the file in read mode
    with open(inputs, 'r') as file:
        # Read each line in the file
        for line in file:
            # Strip the line of leading/trailing spaces and split it
            parts = line.strip().split()

            # If line starts with #, skip it
            if not parts or parts[0].startswith("#"):
                continue
            
            # Ensure there is at least one part, and assign "" if there is no second part
            if len(parts) == 1:
                input_files.append((parts[0], "", 1))  # Only one part, second is an empty string
            elif len(parts) == 2:
                input_files.append((parts[0], parts[1], 1))  # Two parts, normal tuple
            else:
                input_files.append((parts[0], parts[1], parts[2]))

    # Print the list of tuples
    #print(input_files)
    pumpkin_command = f"MZN_SOLVER_PATH=/home/user/Documents/Pumpkin/minizinc minizinc --solver /home/user/Documents/Pumpkin/minizinc/pumpkin.msc --output-time --statistics --output-objective --time-limit {timeout_secs * 1000}"

    commands = [
        ("decomp",f"{pumpkin_command}", []),
        ("regin", f"{pumpkin_command}", [("global_cardinality_low_up(", "pumpkin_gcc_regin("), ("", 'include "pumpkin_gcc.mzn";\n')]),
        ("basic_filter", f"{pumpkin_command}", [("global_cardinality_low_up(", "pumpkin_gcc_basic_filter("), ("", 'include "pumpkin_gcc.mzn";\n')])
    ]  # List of command templates

    #print(commands)

    commands_to_run = []

    # Iterate over each command (or each element in input_files)
    for ifile in input_files:
        repetitions = int(ifile[2])
        for com in commands:

            input_file_path = ifile[0]
            data_file = ifile[1]
            command_name = com[0]
            pumpkin_command = com[1]
            replace_list = com[2]
        
            path, ext = os.path.splitext(os.path.basename(input_file_path))

            # Define the destination path in the temporary directory (using the file name from input_file)
            destination_path = os.path.join(temp_dir, path+f"_{command_name}"+ext)
        
            # Copy the file to the temporary directory, overwriting if the file exists
            shutil.copy(input_file_path, destination_path)

            for replace in replace_list:
                replace_in_file(destination_path, replace[0], replace[1])

            command = f"{pumpkin_command} {destination_path} {data_file}"

            for n in range(repetitions):

                output_path = f"{output_dir}/{os.path.basename(destination_path)}-{os.path.basename(data_file)}-{n}.txt"

                commands_to_run.append(("com[0]", command, output_path))


    # Shuffle benchmarks so repetitions are not always contiguous
    random.seed(1)
    random.shuffle(commands_to_run)

    for _,_,output_path in commands_to_run:
        print(output_path)


    # Generate all combinations of input files, data files, and commands

    # Use ThreadPoolExecutor to run commands on multiple cores
    executor =  ThreadPoolExecutor(max_workers=NUM_CORES)
    futures = []
    for name, cmd, outfile in commands_to_run:
        future = executor.submit(run_command, name, cmd, outfile) 
        futures.append(future)
        #print("future added")

    #[run_command(name, cmd, outfile) for name, cmd, outfile in [commands_to_run[0]]]

        # Track the start time
    start_time = time.time()

    print("done creating futures")
    # Add a callback to each future to print the status when it finishes
    for future in futures:
        future.add_done_callback(lambda f: print_status(futures, start_time, len(futures), commands_to_run))

    # Track progress every 5 seconds
    while True:
        time.sleep(10)  # Wait for 5 seconds
        print_status(futures, start_time, len(futures), commands_to_run)

        if all(future.done() for future in futures):
            break

    
    for i in range(len(futures)):
        future = futures[i]
        _, _, path = commands_to_run[i]
        try:
            future.result()
        except Exception as e:
            print(f"Command '{path}' encountered an error: {e}")

if __name__ == "__main__":
    main()
