#!/usr/bin/env python
# coding: utf-8

# In[44]:


import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

#%matplotlib qtagg


# In[188]:


def extract_runtime(filepath, ignore_unkown_status=False):
    """
    Extracts the last occurrence of the runtime from the file, or returns 0 if the file contains errors.
    """
    runtime_pattern = r"% time elapsed: ([0-9]*\.?[0-9]+) s"
    error_patterns = ["=====UNKNOWN====="] + ["=====ERROR====="]

    if ignore_unkown_status:
        error_patterns = ["=====ERROR====="]

    if not os.path.exists(filepath):
        return 0

    with open(filepath, 'r') as file:
        content = file.read()
        if any(error in content for error in error_patterns):
            return 0

        last_runtime = None
        for line in content.splitlines():
            match = re.search(runtime_pattern, line)
            if match:
                last_runtime = float(match.group(1))

        return last_runtime if last_runtime is not None else 0

def extract_objective(filepath):
    """
    Extracts the last occurrence of the runtime from the file, or returns 0 if the file contains errors.
    """
    objective_pattern = r"objective=([0-9]+)"
    error_patterns = ["=====ERROR====="]

    if not os.path.exists(filepath):
        return 0

    with open(filepath, 'r') as file:
        content = file.read()
        if any(error in content for error in error_patterns):
            return 0

        last_runtime = None
        for line in content.splitlines():
            match = re.search(objective_pattern, line)
            if match:
                last_runtime = float(match.group(1))

        return last_runtime if last_runtime is not None else 0

def parse_benchmark_dirs(directories, ignore_unkown_status=False):
    """
    Parses the directories, extracts runtimes, and groups by problem and technique.

    Args:
        directories (list): List of directories to parse.

    Returns:
        dict: A nested dictionary mapping (problem, data file) to techniques and their runtimes.
    """
    benchmark_data = defaultdict(lambda: defaultdict(list))
    objective_data = defaultdict(lambda: defaultdict(list))

    for directory in directories:
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                # Extract problem name, technique, and data file
                match = re.match(r"(.+?)_(regin|basic_filter|decomp)\.mzn-(.+?)-[0-9]+\.txt", filename)
                if match:
                    problem_name = match.group(1)
                    technique = match.group(2)
                    data_file = match.group(3)
                    filepath = os.path.join(directory, filename)
                    runtime = extract_runtime(filepath, ignore_unkown_status)
                    objective = extract_objective(filepath)

                    benchmark_data[(problem_name, data_file)][technique].append(runtime)
                    objective_data[(problem_name, data_file)][technique].append(objective)

    # Calculate average runtimes for each technique
    avg_runtimes = {
        (problem, data_file): {tech: sum(runtimes) / len(runtimes) if runtimes else 0 for tech, runtimes in techniques.items()}
        for (problem, data_file), techniques in benchmark_data.items()
    }
    avg_objectives = {
        (problem, data_file): {tech: sum(runtimes) / len(runtimes) if runtimes else 0 for tech, runtimes in techniques.items()}
        for (problem, data_file), techniques in objective_data.items()
    }
    return (avg_runtimes, avg_objectives)

def normalize_runtimes(avg_runtimes):
    """
    Normalizes runtimes so that all "decomp" runs are equal to 1.

    Args:
        avg_runtimes (dict): Nested dictionary of problem, data file, and average runtimes by technique.

    Returns:
        dict: Normalized runtimes.
    """
    normalized_runtimes = {}

    for (problem, data_file), tech_runtimes in avg_runtimes.items():
        decomp_runtime = tech_runtimes.get("decomp", 0)
        normalized_runtimes[(problem, data_file)] = {
            tech: (runtime / decomp_runtime if decomp_runtime > 0 else runtime) for tech, runtime in tech_runtimes.items()
        }

    return normalized_runtimes

def abbreviate_text(text, max_length=8):
    """
    Shortens the text to the specified max_length and appends '...' if truncated.

    Args:
        text (str): The text to abbreviate.
        max_length (int): Maximum allowed length of the text.

    Returns:
        str: Abbreviated text.
    """
    return text if len(text) <= max_length else text[:max_length - 3] + '...'

def plot_benchmarks(avg_runtimes, abs_runtimes):
    """
    Creates a grouped bar plot of the normalized runtimes by technique.

    Args:
        avg_runtimes (dict): Nested dictionary of problem, data file, and average runtimes by technique.
    """
    techniques = ["decomp", "regin", "basic_filter"]  # Ensure "decomp" is always first
    colors = {"regin": "steelblue", "basic_filter": "lightblue", "decomp": "gainsboro"}

    # Prepare data for plotting
    grouped_data = defaultdict(list)
    grouped_data_abs = defaultdict(list)
    for (problem, data_file), tech_runtimes in avg_runtimes.items():
        for technique in techniques:
            grouped_data[(problem, data_file)].append(tech_runtimes.get(technique, 0))

    for (problem, data_file), tech_runtimes in abs_runtimes.items():
        for technique in techniques:
            grouped_data_abs[(problem, data_file)].append(tech_runtimes.get(technique, 0))

    # Plotting
    problems = list(grouped_data.keys())
    abbreviated_problems = [f"{abbreviate_text(problem, max_length=10)}, {abbreviate_text(data_file, max_length=20)}" for problem, data_file in problems]
    x = range(len(problems))
    bar_width = 0.2

    plt.figure(figsize=(12, 6))

    # Draw bars and annotate runtimes
    for i, technique in enumerate(techniques):
        runtimes = [group[i] for group in grouped_data.values()]
        runtimes_abs = [group[i] for group in grouped_data_abs.values()]
        bar_positions = [pos + i * bar_width for pos in x]
        plt.bar(
            bar_positions,
            runtimes,
            bar_width,
            label=technique,
            color=colors[technique]
        )
        # Annotate bars with runtimes
        for pos, runtime in zip(bar_positions, runtimes):
            plt.text(pos, runtime + 0.05, f"{runtime:.2f}", ha='center', va='bottom', fontsize=6)

        if technique == "decomp":
            for pos, runtime in zip(bar_positions, runtimes_abs):
                plt.text(pos, -0.3, f"{runtime:.1f}", ha='center', va='bottom', fontsize=8)

    # Draw a baseline for "decomp"
    decomp_positions = [pos + bar_width * 0 for pos in x]  # First position corresponds to "decomp"
    plt.plot(decomp_positions, [1] * len(decomp_positions), '-', label='Baseline (Decomp = 1)', color="black", lw=0.5)

    plt.xlabel('Problems (Problem, Data File)')
    plt.ylabel('Normalized Runtime (Relative to Decomp)')
    plt.title('Normalized Benchmark Runtimes by Technique')
    plt.xticks([pos + bar_width for pos in x], abbreviated_problems, rotation=45, ha='right')
    plt.legend(title="Technique")
    plt.tight_layout()
    plt.show()


# In[190]:


# Replace with your list of directories containing benchmark files
#directories = ["output_evm_super_compilation/", "output_community_detection/", "output_physician_scheduling/", "output_rotating_workforce_scheduling/", "output_vaccine/"]
directories = ["output_evm_super_compilation/", "output_community_detection/", "output_rotating_workforce_scheduling/", "output_community_detection_rnd/" ]

avg_runtimes, avg_objectives = parse_benchmark_dirs(directories)
normalized_runtimes = normalize_runtimes(avg_runtimes)
plot_benchmarks(normalized_runtimes, avg_runtimes)

#normalized_objectives = normalize_runtimes(avg_objectives)
#plot_benchmarks(normalized_objectives, avg_objectives)


# In[183]:


directories = ["output_physician_scheduling-new" ]


avg_runtimes, avg_objectives = parse_benchmark_dirs(directories, ignore_unkown_status=True)
normalized_runtimes = normalize_runtimes(avg_runtimes)
plot_benchmarks(normalized_runtimes, avg_runtimes)

normalized_objectives = avg_objectives
plot_benchmarks(normalized_objectives, avg_objectives)

print(avg_objectives)
print(normalized_objectives)


# In[165]:





# In[167]:





# In[ ]:




