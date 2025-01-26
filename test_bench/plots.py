# %%
import os
import re
import numpy as np
from scipy.stats import gmean
import matplotlib.pyplot as plt
from collections import defaultdict

#%matplotlib qtagg

MAX_RUNTIME = 20*60; # 20 minutes

# %%
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

def extract_lbd(filepath):
    """
    Extracts the last occurrence of the LBD from the file, or returns 0 if the file contains errors.
    """
    lbd_pattern = r"learnedClauseStatisticsAverageLbd=([0-9]*\.?[0-9]+)"
    error_patterns = ["=====ERROR====="]

    if not os.path.exists(filepath):
        return 0

    with open(filepath, 'r') as file:
        content = file.read()
        if any(error in content for error in error_patterns):
            return 0

        last_lbd = None
        for line in content.splitlines():
            match = re.search(lbd_pattern, line)
            if match:
                last_lbd = float(match.group(1))

        return last_lbd if last_lbd is not None else 0

def extract_learned_clause_length(filepath):
    """
    Extracts the last occurrence of the learned clause length from the file, or returns 0 if the file contains errors.
    """
    learned_clause_length_pattern = r"learnedClauseStatisticsAverageLearnedClauseLength=([0-9]*\.?[0-9]+)"
    error_patterns = ["=====ERROR====="]

    if not os.path.exists(filepath):
        return 0

    with open(filepath, 'r') as file:
        content = file.read()
        if any(error in content for error in error_patterns):
            return 0

        last_learned_clause_length = None
        for line in content.splitlines():
            match = re.search(learned_clause_length_pattern, line)
            if match:
                last_learned_clause_length = float(match.group(1))

        return last_learned_clause_length if last_learned_clause_length is not None else 0

def extract_conflict_size(filepath):
    """
    Extracts the last occurrence of the average conflict size from the file, or returns 0 if the file contains errors.
    """
    conflict_size_pattern = r"learnedClauseStatisticsAverageConflictSize=([0-9]*\.?[0-9]+)"
    error_patterns = ["=====ERROR====="]

    if not os.path.exists(filepath):
        return 0

    with open(filepath, 'r') as file:
        content = file.read()
        if any(error in content for error in error_patterns):
            return 0

        last_conflict_size = None
        for line in content.splitlines():
            match = re.search(conflict_size_pattern, line)
            if match:
                last_conflict_size = float(match.group(1))

        return last_conflict_size if last_conflict_size is not None else 0

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
    lbd_data = defaultdict(lambda: defaultdict(list))
    learned_clause_length_data = defaultdict(lambda: defaultdict(list))
    conflict_size_data = defaultdict(lambda: defaultdict(list))

    for directory in directories:
        for filename in os.listdir(directory):
            if filename.endswith('.txt') and not filename.startswith("#"):
                # Extract problem name, technique, and data file
                match = re.match(r"(.+?)_(regin|basic_filter|decomp)\.mzn-(.+?)-[0-9]+\.txt", filename)
                if match:
                    problem_name = match.group(1)
                    technique = match.group(2)
                    data_file = match.group(3)
                    filepath = os.path.join(directory, filename)
                    runtime = extract_runtime(filepath, ignore_unkown_status)
                    objective = extract_objective(filepath)
                    lbd = extract_lbd(filepath)
                    learned_clause_length = extract_learned_clause_length(filepath)
                    conflict_size = extract_conflict_size(filepath)

                    benchmark_data[(problem_name, data_file)][technique].append(runtime)
                    objective_data[(problem_name, data_file)][technique].append(objective)
                    lbd_data[(problem_name, data_file)][technique].append(lbd)
                    learned_clause_length_data[(problem_name, data_file)][technique].append(learned_clause_length)
                    conflict_size_data[(problem_name, data_file)][technique].append(conflict_size)

    # Calculate average runtimes for each technique
    avg_runtimes = {
        (problem, data_file): {tech: sum(runtimes) / len(runtimes) if runtimes else 0 for tech, runtimes in techniques.items()}
        for (problem, data_file), techniques in benchmark_data.items()
    }

    standard_deviation_runtimes = {
        (problem, data_file): {tech: np.std(runtimes) if runtimes else 0 for tech, runtimes in techniques.items()}
        for (problem, data_file), techniques in benchmark_data.items()
    }
    avg_objectives = {
        (problem, data_file): {tech: sum(runtimes) / len(runtimes) if runtimes else 0 for tech, runtimes in techniques.items()}
        for (problem, data_file), techniques in objective_data.items()
    }
    avg_lbds = {
        (problem, data_file): {tech: sum(lbds) / len(lbds) if lbds else 0 for tech, lbds in techniques.items()}
        for (problem, data_file), techniques in lbd_data.items()
    }
    avg_learned_clause_lengths = {
        (problem, data_file): {tech: sum(lengths) / len(lengths) if lengths else 0 for tech, lengths in techniques.items()}
        for (problem, data_file), techniques in learned_clause_length_data.items()}
    avg_conflict_sizes = {
        (problem, data_file): {tech: sum(sizes) / len(sizes) if sizes else 0 for tech, sizes in techniques.items()}
        for (problem, data_file), techniques in conflict_size_data.items()
    }
    return (avg_runtimes, avg_objectives, avg_lbds, avg_learned_clause_lengths, avg_conflict_sizes, standard_deviation_runtimes)

def normalize(avg_values):
    """
    Normalizes values so that all "decomp" runs are equal to 1.

    Args:
        avg_values (dict): Nested dictionary of problem, data file, and average values by technique.

    Returns:
        dict: Normalized values.
    """
    normalized_values = {}

    for (problem, data_file), tech_values in avg_values.items():
        decomp_value = tech_values.get("decomp", 0)
        normalized_values[(problem, data_file)] = {
            tech: (value / decomp_value if decomp_value > 0 else value / MAX_RUNTIME) for tech, value in tech_values.items()
        }

    return normalized_values

def normalize_objective(avg_values):
    """
    Normalizes values so that all "decomp" runs are equal to 1.

    Args:
        avg_values (dict): Nested dictionary of problem, data file, and average values by technique.

    Returns:
        dict: Normalized values.
    """
    normalized_values = {}

    for (problem, data_file), tech_values in avg_values.items():
        decomp_value = tech_values.get("decomp", 0)
        normalized_values[(problem, data_file)] = {
            tech: (value / decomp_value if decomp_value > 0 else (2 if value > 0.0001 else 0)) for tech, value in tech_values.items()
        }

    return normalized_values

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

def plot_benchmarks(avg_values, abs_values, title='Normalized Benchmark Values by Technique', hide_problem_name=False):
    """
    Creates a grouped bar plot of the normalized values by technique.

    Args:
        avg_values (dict): Nested dictionary of problem, data file, and average values by technique.
        abs_values (dict): Nested dictionary of problem, data file, and absolute values by technique.
        title (str): Title of the plot.
    """
    techniques = ["decomp", "regin", "basic_filter"]  # Ensure "decomp" is always first
    colors = {"regin": "steelblue", "basic_filter": "lightblue", "decomp": "gainsboro"}

    # Prepare data for plotting
    grouped_data = defaultdict(list)
    grouped_data_abs = defaultdict(list)
    for (problem, data_file), tech_values in avg_values.items():
        for technique in techniques:
            grouped_data[(problem, data_file)].append(tech_values.get(technique, 0))

    for (problem, data_file), tech_values in abs_values.items():
        for technique in techniques:
            grouped_data_abs[(problem, data_file)].append(tech_values.get(technique, 0))

    # Sort problems alphabetically by their label
    problems = sorted(grouped_data.keys(), key=lambda x: f"{x[0]}, {x[1]}")
    abbreviated_problems = [f"{str(abbreviate_text(problem, max_length=10))+", " if not hide_problem_name else ""}{abbreviate_text(data_file, max_length=20)}" for problem, data_file in problems]
    x = range(len(problems))
    bar_width = 0.2

    plt.figure(figsize=(12, 6))

    # Draw bars and annotate values
    for i, technique in enumerate(techniques):
        values = [grouped_data[(problem, data_file)][i] for problem, data_file in problems]
        values_abs = [grouped_data_abs[(problem, data_file)][i] for problem, data_file in problems]
        bar_positions = [pos + i * bar_width for pos in x]
        plt.bar(
            bar_positions,
            values,
            bar_width,
            label=technique,
            color=colors[technique]
        )

        # Annotate bars with values
        for pos, value, value_abs in zip(bar_positions, values, values_abs):
            if (technique == "regin" or technique == "basic_filter") and value > 0.000001:
                plt.text(pos, value + 0.15, f"{value:.2f}", ha='center', va='bottom', fontsize=4)
            #if (technique == "regin" or technique == "basic_filter") and value_abs > 0.000001:
            #    plt.text(pos, 0.01, f"{value_abs:.2f}", ha='center', va='bottom', fontsize=4, rotation=90, color='black')

    # Draw a baseline for "decomp"
    decomp_positions = [pos + bar_width * 0 for pos in x]  # First position corresponds to "decomp"
    plt.plot(decomp_positions, [1] * len(decomp_positions), '-', label='Baseline (Decomp = 1)', color="black", lw=0.5)

    plt.xlabel('Problems (Problem, Data File)')
    plt.ylabel('Normalized Value (Relative to Decomp)')
    plt.title(title)
    plt.xticks([pos + bar_width for pos in x], abbreviated_problems, rotation=45, ha='right', fontsize=5)
    plt.legend(title="Technique")
    plt.yscale('symlog')  # Set y-axis to symmetric logarithmic scale
    plt.tight_layout()
    plt.show()

def plot_all_statistics(avg_runtimes, avg_lbds, avg_learned_clause_lengths, avg_conflict_sizes, abs_runtimes, abs_lbds, abs_learned_clause_lengths, abs_conflict_sizes):
    """
    Creates scatter plots of benchmark statistics (LBD, learned clause length, and conflict size) by technique.

    Args:
        avg_runtimes (dict): Normalized runtime values for plotting.
        avg_lbds (dict): Normalized LBD values for plotting.
        avg_learned_clause_lengths (dict): Normalized learned clause length values for plotting.
        avg_conflict_sizes (dict): Normalized conflict size values for plotting.
        abs_runtimes (dict): Absolute runtime values for annotation.
        abs_lbds (dict): Absolute LBD values for annotation.
        abs_learned_clause_lengths (dict): Absolute learned clause length values for annotation.
        abs_conflict_sizes (dict): Absolute conflict size values for annotation.
    """
    techniques = ["decomp", "regin", "basic_filter"]  # Ensure "decomp" is always first
    colors = {"regin": "steelblue", "basic_filter": "lightblue", "decomp": "gainsboro"}
    markers = {"regin": "o", "basic_filter": "s", "decomp": "D"}  # Different markers for techniques

    # Prepare data for plotting
    grouped_data_lbd = defaultdict(list)
    grouped_data_learned_clause_length = defaultdict(list)
    grouped_data_conflict_size = defaultdict(list)
    grouped_data_abs_lbd = defaultdict(list)
    grouped_data_abs_learned_clause_length = defaultdict(list)
    grouped_data_abs_conflict_size = defaultdict(list)

    for (problem, data_file), tech_lbds in avg_lbds.items():
        for technique in techniques:
            grouped_data_lbd[(problem, data_file)].append(tech_lbds.get(technique, 0))

    for (problem, data_file), tech_learned_clause_lengths in avg_learned_clause_lengths.items():
        for technique in techniques:
            grouped_data_learned_clause_length[(problem, data_file)].append(tech_learned_clause_lengths.get(technique, 0))

    for (problem, data_file), tech_conflict_sizes in avg_conflict_sizes.items():
        for technique in techniques:
            grouped_data_conflict_size[(problem, data_file)].append(tech_conflict_sizes.get(technique, 0))

    for (problem, data_file), tech_lbds in abs_lbds.items():
        for technique in techniques:
            grouped_data_abs_lbd[(problem, data_file)].append(tech_lbds.get(technique, 0))

    for (problem, data_file), tech_learned_clause_lengths in abs_learned_clause_lengths.items():
        for technique in techniques:
            grouped_data_abs_learned_clause_length[(problem, data_file)].append(tech_learned_clause_lengths.get(technique, 0))

    for (problem, data_file), tech_conflict_sizes in abs_conflict_sizes.items():
        for technique in techniques:
            grouped_data_abs_conflict_size[(problem, data_file)].append(tech_conflict_sizes.get(technique, 0))

    # Sort problems alphabetically by their label
    problems = sorted(grouped_data_lbd.keys(), key=lambda x: f"{x[0]}, {x[1]}")
    abbreviated_problems = [f"{abbreviate_text(problem, max_length=10)}, {abbreviate_text(data_file, max_length=20)}" for problem, data_file in problems]
    x = range(len(problems))
    bar_width = 0.2

    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]})

    # Define decomp_positions
    decomp_positions = [pos + bar_width * 0 for pos in x]  # First position corresponds to "decomp"

    # Plot LBD scatter plot
    for i, technique in enumerate(techniques):
        lbds = [grouped_data_lbd[(problem, data_file)][i] for problem, data_file in problems]
        lbds_abs = [grouped_data_abs_lbd[(problem, data_file)][i] for problem, data_file in problems]
        x_positions = [pos + i * bar_width for pos in x]
        axs[0].scatter(
            x_positions,
            lbds,
            label=f'{technique} (LBD)',
            color=colors[technique],
            marker=markers[technique],
            edgecolor="black",
            s=25  # Marker size
        )
        # Annotate points with LBDs
        for pos, lbd, abs_lbd in zip(x_positions, lbds, lbds_abs):
            if (technique == "regin" or technique == "basic_filter") and lbd > 0.01:
                axs[0].text(pos, lbd + 0.1, f"{lbd:.2f}", ha='center', va='bottom', fontsize=6)
            if technique == "decomp":  # Additional annotation for "decomp"
                axs[0].text(pos, 0, f"{abs_lbd:.1f}", ha='center', va='top', fontsize=6)

        # Draw lines connecting points
        if technique in ["regin", "basic_filter"]:
            axs[0].plot(x_positions, lbds, color=colors[technique], linestyle='-', linewidth=1)

    # Draw a baseline for "decomp" on LBD axis
    axs[0].plot(decomp_positions, [1] * len(decomp_positions), '-', label='Baseline (Decomp = 1)', color="black", lw=0.5)

    axs[0].set_ylabel('Normalized LBD')
    axs[0].set_title('Normalized LBD by Technique (Relative to Decomp)')
    axs[0].legend(title="Technique")
    axs[0].set_yscale('symlog')  # Set y-axis to symmetric logarithmic scale

    # Plot learned clause length scatter plot
    for i, technique in enumerate(techniques):
        learned_clause_lengths = [grouped_data_learned_clause_length[(problem, data_file)][i] for problem, data_file in problems]
        learned_clause_lengths_abs = [grouped_data_abs_learned_clause_length[(problem, data_file)][i] for problem, data_file in problems]
        x_positions = [pos + i * bar_width for pos in x]
        axs[1].scatter(
            x_positions,
            learned_clause_lengths,
            label=f'{technique} (Learned Clause Length)',
            color=colors[technique],
            marker=markers[technique],
            edgecolor="black",
            s=25  # Marker size
        )
        # Annotate points with learned clause lengths
        for pos, length, abs_length in zip(x_positions, learned_clause_lengths, learned_clause_lengths_abs):
            if (technique == "regin" or technique == "basic_filter") and length > 0.01:
                axs[1].text(pos, length + 0.1, f"{length:.2f}", ha='center', va='bottom', fontsize=6)
            if technique == "decomp":  # Additional annotation for "decomp"
                axs[1].text(pos, 0, f"{abs_length:.1f}", ha='center', va='top', fontsize=6)

        # Draw lines connecting points
        if technique in ["regin", "basic_filter"]:
            axs[1].plot(x_positions, learned_clause_lengths, color=colors[technique], linestyle='-', linewidth=1)

    # Draw a baseline for "decomp" on learned clause length axis
    axs[1].plot(decomp_positions, [1] * len(decomp_positions), '-', label='Baseline (Decomp = 1)', color="black", lw=0.5)

    axs[1].set_ylabel('Normalized Learned Clause Length')
    axs[1].set_title('Normalized Learned Clause Length by Technique (Relative to Decomp)')
    axs[1].legend(title="Technique")
    axs[1].set_yscale('symlog')  # Set y-axis to symmetric logarithmic scale

    # Plot conflict size scatter plot
    for i, technique in enumerate(techniques):
        conflict_sizes = [grouped_data_conflict_size[(problem, data_file)][i] for problem, data_file in problems]
        conflict_sizes_abs = [grouped_data_abs_conflict_size[(problem, data_file)][i] for problem, data_file in problems]
        x_positions = [pos + i * bar_width for pos in x]
        axs[2].scatter(
            x_positions,
            conflict_sizes,
            label=f'{technique} (Conflict Size)',
            color=colors[technique],
            marker=markers[technique],
            edgecolor="black",
            s=25  # Marker size
        )
        # Annotate points with conflict sizes
        for pos, size, abs_size in zip(x_positions, conflict_sizes, conflict_sizes_abs):
            if (technique == "regin" or technique == "basic_filter") and size > 0.01:
                axs[2].text(pos, size + 0.1, f"{size:.2f}", ha='center', va='bottom', fontsize=6)
            if technique == "decomp":  # Additional annotation for "decomp"
                axs[2].text(pos, 0, f"{abs_size:.1f}", ha='center', va='top', fontsize=6)

        # Draw lines connecting points
        if technique in ["regin", "basic_filter"]:
            axs[2].plot(x_positions, conflict_sizes, color=colors[technique], linestyle='-', linewidth=1)

    # Draw a baseline for "decomp" on conflict size axis
    axs[2].plot(decomp_positions, [1] * len(decomp_positions), '-', label='Baseline (Decomp = 1)', color="black", lw=0.5)

    axs[2].set_xlabel('Problems (Problem, Data File)')
    axs[2].set_ylabel('Normalized Conflict Size')
    axs[2].set_title('Normalized Conflict Size by Technique (Relative to Decomp)')
    axs[2].legend(title="Technique")
    axs[2].set_yscale('symlog')  # Set y-axis to symmetric logarithmic scale

    axs[0].set_xticks([pos + bar_width for pos in x])
    axs[2].set_xticklabels(abbreviated_problems, rotation=45, ha='right')

    fig.tight_layout()
    plt.show()

def plot_all_statistics_bar(avg_runtimes, avg_lbds, avg_learned_clause_lengths, avg_conflict_sizes, abs_runtimes, abs_lbds, abs_learned_clause_lengths, abs_conflict_sizes):
    """
    Creates bar plots of benchmark statistics (runtime, LBD, learned clause length, and conflict size) by technique.

    Args:
        avg_runtimes (dict): Normalized runtime values for plotting.
        avg_lbds (dict): Normalized LBD values for plotting.
        avg_learned_clause_lengths (dict): Normalized learned clause length values for plotting.
        avg_conflict_sizes (dict): Normalized conflict size values for plotting.
        abs_runtimes (dict): Absolute runtime values for annotation.
        abs_lbds (dict): Absolute LBD values for annotation.
        abs_learned_clause_lengths (dict): Absolute learned clause length values for annotation.
        abs_conflict_sizes (dict): Absolute conflict size values for annotation.
    """
    techniques = ["decomp", "regin", "basic_filter"]  # Ensure "decomp" is always first
    colors = {"regin": "steelblue", "basic_filter": "lightblue", "decomp": "gainsboro"}

    # Prepare data for plotting
    grouped_data_runtime = defaultdict(list)
    grouped_data_lbd = defaultdict(list)
    grouped_data_learned_clause_length = defaultdict(list)
    grouped_data_conflict_size = defaultdict(list)
    grouped_data_abs_runtime = defaultdict(list)
    grouped_data_abs_lbd = defaultdict(list)
    grouped_data_abs_learned_clause_length = defaultdict(list)
    grouped_data_abs_conflict_size = defaultdict(list)

    for (problem, data_file), tech_runtimes in avg_runtimes.items():
        for technique in techniques:
            grouped_data_runtime[(problem, data_file)].append(tech_runtimes.get(technique, 0))

    for (problem, data_file), tech_lbds in avg_lbds.items():
        for technique in techniques:
            grouped_data_lbd[(problem, data_file)].append(tech_lbds.get(technique, 0))

    for (problem, data_file), tech_learned_clause_lengths in avg_learned_clause_lengths.items():
        for technique in techniques:
            grouped_data_learned_clause_length[(problem, data_file)].append(tech_learned_clause_lengths.get(technique, 0))

    for (problem, data_file), tech_conflict_sizes in avg_conflict_sizes.items():
        for technique in techniques:
            grouped_data_conflict_size[(problem, data_file)].append(tech_conflict_sizes.get(technique, 0))

    for (problem, data_file), tech_runtimes in abs_runtimes.items():
        for technique in techniques:
            grouped_data_abs_runtime[(problem, data_file)].append(tech_runtimes.get(technique, 0))

    for (problem, data_file), tech_lbds in abs_lbds.items():
        for technique in techniques:
            grouped_data_abs_lbd[(problem, data_file)].append(tech_lbds.get(technique, 0))

    for (problem, data_file), tech_learned_clause_lengths in abs_learned_clause_lengths.items():
        for technique in techniques:
            grouped_data_abs_learned_clause_length[(problem, data_file)].append(tech_learned_clause_lengths.get(technique, 0))

    for (problem, data_file), tech_conflict_sizes in abs_conflict_sizes.items():
        for technique in techniques:
            grouped_data_abs_conflict_size[(problem, data_file)].append(tech_conflict_sizes.get(technique, 0))

    # Sort problems alphabetically by their label
    problems = sorted(grouped_data_runtime.keys(), key=lambda x: f"{x[0]}, {x[1]}")
    abbreviated_problems = [f"{abbreviate_text(problem, max_length=10)}, {abbreviate_text(data_file, max_length=20)}" for problem, data_file in problems]
    x = range(len(problems))
    bar_width = 0.2

    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1]})

    # Plot runtime bar chart
    for i, technique in enumerate(techniques):
        runtimes = [grouped_data_runtime[(problem, data_file)][i] for problem, data_file in problems]
        runtimes_abs = [grouped_data_abs_runtime[(problem, data_file)][i] for problem, data_file in problems]
        bar_positions = [pos + i * bar_width for pos in x]
        axs[0].bar(
            bar_positions,
            runtimes,
            bar_width,
            label=technique,
            color=colors[technique]
        )
        # Annotate bars with runtimes
        for pos, runtime, abs_runtime in zip(bar_positions, runtimes, runtimes_abs):
            axs[0].text(pos, runtime + 0.05, f"{runtime:.2f}", ha='center', va='bottom', fontsize=6)
            if technique == "decomp":  # Additional annotation for "decomp"
                axs[0].text(pos, runtime / 2, f"{abs_runtime:.1f}", ha='center', va='center', fontsize=8, rotation=90)

    # Draw a baseline for "decomp"
    decomp_positions = [pos + bar_width * 0 for pos in x]  # First position corresponds to "decomp"
    axs[0].plot(decomp_positions, [1] * len(decomp_positions), '-', label='Baseline (Decomp = 1)', color="black", lw=0.5)

    axs[0].set_ylabel('Normalized Runtime (Relative to Decomp)')
    axs[0].set_title('Normalized Benchmark Runtimes by Technique')
    axs[0].legend(title="Technique")
    axs[0].set_yscale('symlog')  # Set y-axis to symmetric logarithmic scale

    # Plot LBD bar chart
    for i, technique in enumerate(techniques):
        lbds = [grouped_data_lbd[(problem, data_file)][i] for problem, data_file in problems]
        lbds_abs = [grouped_data_abs_lbd[(problem, data_file)][i] for problem, data_file in problems]
        bar_positions = [pos + i * bar_width for pos in x]
        axs[1].bar(
            bar_positions,
            lbds,
            bar_width,
            label=technique,
            color=colors[technique]
        )
        # Annotate bars with LBDs
        for pos, lbd, abs_lbd in zip(bar_positions, lbds, lbds_abs):
            axs[1].text(pos, lbd + 0.05, f"{lbd:.2f}", ha='center', va='bottom', fontsize=6, alpha=0.5)
            if technique == "decomp":  # Additional annotation for "decomp"
                axs[1].text(pos, lbd / 2, f"{abs_lbd:.1f}", ha='center', va='center', fontsize=8, rotation=90, alpha=0.5)

    # Draw a baseline for "decomp" on LBD axis
    axs[1].plot(decomp_positions, [1] * len(decomp_positions), '-', label='Baseline (Decomp = 1)', color="black", lw=0.5)

    axs[1].set_ylabel('Normalized LBD (Relative to Decomp)')
    axs[1].set_title('Normalized Benchmark LBD by Technique')
    axs[1].legend(title="Technique")
    axs[1].set_yscale('symlog')  # Set y-axis to symmetric logarithmic scale

    # Plot learned clause length bar chart
    for i, technique in enumerate(techniques):
        learned_clause_lengths = [grouped_data_learned_clause_length[(problem, data_file)][i] for problem, data_file in problems]
        learned_clause_lengths_abs = [grouped_data_abs_learned_clause_length[(problem, data_file)][i] for problem, data_file in problems]
        bar_positions = [pos + i * bar_width for pos in x]
        axs[2].bar(
            bar_positions,
            learned_clause_lengths,
            bar_width,
            label=technique,
            color=colors[technique]
        )
        # Annotate bars with learned clause lengths
        for pos, length, abs_length in zip(bar_positions, learned_clause_lengths, learned_clause_lengths_abs):
            axs[2].text(pos, length + 0.05, f"{length:.2f}", ha='center', va='bottom', fontsize=6, alpha=0.5)
            if technique == "decomp":  # Additional annotation for "decomp"
                axs[2].text(pos, length / 2, f"{abs_length:.1f}", ha='center', va='center', fontsize=8, rotation=90, alpha=0.5)

    # Draw a baseline for "decomp" on learned clause length axis
    axs[2].plot(decomp_positions, [1] * len(decomp_positions), '-', label='Baseline (Decomp = 1)', color="black", lw=0.5)

    axs[2].set_ylabel('Normalized Learned Clause Length (Relative to Decomp)')
    axs[2].set_title('Normalized Benchmark Learned Clause Length by Technique')
    axs[2].legend(title="Technique")
    axs[2].set_yscale('symlog')  # Set y-axis to symmetric logarithmic scale

    # Plot conflict size bar chart
    for i, technique in enumerate(techniques):
        conflict_sizes = [grouped_data_conflict_size[(problem, data_file)][i] for problem, data_file in problems]
        conflict_sizes_abs = [grouped_data_abs_conflict_size[(problem, data_file)][i] for problem, data_file in problems]
        bar_positions = [pos + i * bar_width for pos in x]
        axs[3].bar(
            bar_positions,
            conflict_sizes,
            bar_width,
            label=technique,
            color=colors[technique]
        )
        # Annotate bars with conflict sizes
        for pos, size, abs_size in zip(bar_positions, conflict_sizes, conflict_sizes_abs):
            axs[3].text(pos, size + 0.05, f"{size:.2f}", ha='center', va='bottom', fontsize=6, alpha=0.5)
            if technique == "decomp":  # Additional annotation for "decomp"
                axs[3].text(pos, size / 2, f"{abs_size:.1f}", ha='center', va='center', fontsize=8, rotation=90, alpha=0.5)

    # Draw a baseline for "decomp" on conflict size axis
    axs[3].plot(decomp_positions, [1] * len(decomp_positions), '-', label='Baseline (Decomp = 1)', color="black", lw=0.5)

    axs[3].set_xlabel('Problems (Problem, Data File)')
    axs[3].set_ylabel('Normalized Conflict Size (Relative to Decomp)')
    axs[3].set_title('Normalized Benchmark Conflict Size by Technique')
    axs[3].legend(title="Technique")
    axs[3].set_yscale('symlog')  # Set y-axis to symmetric logarithmic scale

    axs[0].set_xticks([pos + bar_width for pos in x])
    axs[3].set_xticklabels(abbreviated_problems, rotation=45, ha='right')

    fig.tight_layout()
    plt.show()

def plot_runtime_and_objective(avg_runtimes, avg_objectives, abs_runtimes, abs_objectives):
    """
    Creates a combined bar plot of runtime and objective values by technique.

    Args:
        avg_runtimes (dict): Normalized runtime values for plotting.
        avg_objectives (dict): Normalized objective values for plotting.
        abs_runtimes (dict): Absolute runtime values for annotation.
        abs_objectives (dict): Absolute objective values for annotation.
    """
    techniques = ["decomp", "regin", "basic_filter"]  # Ensure "decomp" is always first
    colors = {"regin": "steelblue", "basic_filter": "lightblue", "decomp": "gainsboro"}

    # Prepare data for plotting
    grouped_data_runtime = defaultdict(list)
    grouped_data_objective = defaultdict(list)
    grouped_data_abs_runtime = defaultdict(list)
    grouped_data_abs_objective = defaultdict(list)

    for (problem, data_file), tech_runtimes in avg_runtimes.items():
        for technique in techniques:
            grouped_data_runtime[(problem, data_file)].append(tech_runtimes.get(technique, 0))

    for (problem, data_file), tech_objectives in avg_objectives.items():
        for technique in techniques:
            grouped_data_objective[(problem, data_file)].append(tech_objectives.get(technique, 0))

    for (problem, data_file), tech_runtimes in abs_runtimes.items():
        for technique in techniques:
            grouped_data_abs_runtime[(problem, data_file)].append(tech_runtimes.get(technique, 0))

    for (problem, data_file), tech_objectives in abs_objectives.items():
        for technique in techniques:
            grouped_data_abs_objective[(problem, data_file)].append(tech_objectives.get(technique, 0))

    # Sort problems alphabetically by their label
    problems = sorted(grouped_data_runtime.keys(), key=lambda x: f"{x[0]}, {x[1]}")
    abbreviated_problems = [f"{abbreviate_text(problem, max_length=10)}, {abbreviate_text(data_file, max_length=20)}" for problem, data_file in problems]
    x = range(len(problems))
    bar_width = 0.2

    fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

    # Plot runtime bar chart
    for i, technique in enumerate(techniques):
        runtimes = [grouped_data_runtime[(problem, data_file)][i] for problem, data_file in problems]
        runtimes_abs = [grouped_data_abs_runtime[(problem, data_file)][i] for problem, data_file in problems]
        bar_positions = [pos + i * bar_width for pos in x]
        axs[0].bar(
            bar_positions,
            runtimes,
            bar_width,
            label=technique,
            color=colors[technique]
        )
        # Annotate bars with runtimes
        for pos, runtime, abs_runtime in zip(bar_positions, runtimes, runtimes_abs):
            if (technique == "regin" or technique == "basic_filter") and runtime > 0.01:
                axs[0].text(pos, runtime + 0.15 , f"{runtime:.2f}", ha='center', va='bottom', fontsize=5)
            
            if abs_runtime > 0.1:
                axs[0].text(pos, 0.01, f"{abs_runtime:.1f}", ha='center', va='bottom', fontsize=5, rotation=90, color='black', fontweight='bold')


    # Draw a baseline for "decomp"
    decomp_positions = [pos + bar_width * 0 for pos in x]  # First position corresponds to "decomp"
    axs[0].plot(decomp_positions, [1] * len(decomp_positions), '-', label='Baseline (Decomp = 1)', color="black", lw=0.5)

    axs[0].set_ylabel('Normalized Runtime (Relative to Decomp)')
    axs[0].set_title('Normalized Benchmark Runtimes by Technique')
    axs[0].legend(title="Technique")
    axs[0].set_yscale('symlog')  # Set y-axis to symmetric logarithmic scale

    # Plot objective bar chart
    for i, technique in enumerate(techniques):
        objectives = [grouped_data_objective[(problem, data_file)][i] for problem, data_file in problems]
        objectives_abs = [grouped_data_abs_objective[(problem, data_file)][i] for problem, data_file in problems]
        bar_positions = [pos + i * bar_width for pos in x]
        axs[1].bar(
            bar_positions,
            objectives,
            bar_width,
            label=technique,
            color=colors[technique]
        )
        # Annotate bars with objectives
        for pos, objective, abs_objective in zip(bar_positions, objectives, objectives_abs):
            
            if abs_objective > 0.1:
                axs[1].text(pos, 0.01, f"{abs_objective:.1f}", ha='center', va='bottom', fontsize=5, rotation=90, color='black', fontweight='bold')

    # Draw a baseline for "decomp" on objective axis
    axs[1].plot(decomp_positions, [1] * len(decomp_positions), '-', label='Baseline (Decomp = 1)', color="black", lw=0.5)

    axs[1].set_ylabel('Normalized Objective Value (Relative to Decomp)')
    axs[1].set_title('Normalized Benchmark Objective Values by Technique')
    axs[1].legend(title="Technique")
    axs[1].set_yscale('symlog')  # Set y-axis to symmetric logarithmic scale

    axs[1].set_xlabel('Problems (Problem, Data File)')
    axs[1].set_xticks([pos + bar_width for pos in x])
    axs[1].set_xticklabels(abbreviated_problems, rotation=45, ha='right')

    fig.tight_layout()
    plt.show()

def print_standard_deviation(avg_runtimes, std):
    """
    Prints the average runtime and standard deviation of runtimes for each technique in a readable format.

    Args:
        avg_runtimes (dict): Nested dictionary of problem, data file, and average runtimes by technique.
        std (dict): Nested dictionary of problem, data file, and standard deviation of runtimes by technique.
    """
    for (problem, data_file), tech_std in std.items():
        print(f"Problem: {problem}, Data File: {data_file}")
        for technique, deviation in tech_std.items():
            avg_runtime = avg_runtimes[(problem, data_file)].get(technique, 0)
            print(f"  Technique: {technique}, Average Runtime: {avg_runtime:.2f}, Standard Deviation: {deviation:.2f}")

def print_mean_ratios_and_success_rate(avg_runtimes, stat_type="Runtime", remove_missing=False, invert=False, max_runtime=MAX_RUNTIME):
    """
    Prints the mean of ratios of the runtimes of Regin vs Decomp and Basic Filter vs Decomp.
    Considers timeouts and reports the success rate for each technique.

    Args:
        avg_runtimes (dict): Nested dictionary of problem, data file, and average runtimes by technique.
    """
    regin_ratios = []
    basic_filter_ratios = []
    regin_did_not_terminate = 0
    basic_filter_did_not_terminate = 0
    decomp_did_not_terminate = 0
    total_instances = 0

    for tech_runtimes in avg_runtimes.values():
        #print(tech_runtimes)
        decomp_runtime = tech_runtimes.get("decomp", 0)
        regin_runtime = tech_runtimes.get("regin", 0)
        basic_filter_runtime = tech_runtimes.get("basic_filter", 0)
        #print(decomp_runtime, regin_runtime, basic_filter_runtime)

        if remove_missing and (decomp_runtime == 0 or regin_runtime == 0 or basic_filter_runtime == 0):
            continue

        if decomp_runtime == 0:
            decomp_did_not_terminate += 1
            decomp_runtime = max_runtime
            #print("Decomp did not terminate")

        if regin_runtime == 0:
            regin_did_not_terminate += 1
            regin_runtime = max_runtime
            #print("Regin did not terminate")

        if basic_filter_runtime == 0:
            basic_filter_did_not_terminate += 1
            basic_filter_runtime = max_runtime
            #print("Basic Filter did not terminate")

        if not (regin_runtime == 0 and decomp_runtime == 0):
            #print("REG: ",regin_runtime, decomp_runtime, decomp_runtime / regin_runtime if invert else regin_runtime / decomp_runtime)
            regin_ratios.append(decomp_runtime / regin_runtime if invert else regin_runtime / decomp_runtime)

        

        if not (basic_filter_runtime == 0 and decomp_runtime == 0):
            #print("BF: ", basic_filter_runtime, decomp_runtime, decomp_runtime / basic_filter_runtime if invert else basic_filter_runtime / decomp_runtime)
            basic_filter_ratios.append(decomp_runtime / basic_filter_runtime if invert else basic_filter_runtime / decomp_runtime)

        total_instances += 1

    mean_regin_ratio = np.mean(regin_ratios) if regin_ratios else float('inf')
    #print(f"regin ratios: {regin_ratios}");
    #print(f"basic filter ratios: {basic_filter_ratios}");
    mean_basic_filter_ratio = np.mean(basic_filter_ratios) if basic_filter_ratios else float('inf')

    geometric_mean_regin_ratio = gmean(regin_ratios) if regin_ratios else float('inf')
    geometric_mean_basic_filter_ratio = gmean(basic_filter_ratios) if basic_filter_ratios else float('inf')

    regin_success_rate = (total_instances - regin_did_not_terminate) / total_instances
    basic_filter_success_rate = (total_instances - basic_filter_did_not_terminate) / total_instances
    decomp_success_rate = (total_instances - decomp_did_not_terminate) / total_instances

    print(f"Mean {stat_type} Ratio (Regin vs Decomp) {"(invert)" if invert else ""}: arithmetic: {mean_regin_ratio:.2f}, geometric: {geometric_mean_regin_ratio:.2f}")
    print(f"Mean {stat_type} Ratio (Basic Filter vs Decomp) {"(invert)" if invert else ""}: arithmetic: {mean_basic_filter_ratio:.2f}, geometric: {geometric_mean_basic_filter_ratio:.2f}")

    if not remove_missing:
        print(f"Success Rate (Regin): {regin_success_rate:.2%} (Basic Filter): {basic_filter_success_rate:.2%} (Decomp): {decomp_success_rate:.2%}")


# %%
# Replace with your list of directories containing benchmark files
#directories = ["output_evm_super_compilation/", "output_community_detection/", "output_physician_scheduling/", "output_rotating_workforce_scheduling/", "output_vaccine/"]
#directories = ["output_evm_super_compilation/", "output_community_detection/", "output_rotating_workforce_scheduling/", "output_community_detection_rnd/" ]
directories = ["output_all-new/" ]

avg_runtimes, avg_objectives, avg_lbds, avg_learned_clause_lengths, avg_conflict_sizes, std = parse_benchmark_dirs(directories)
normalized_runtimes = normalize(avg_runtimes)
normalized_objectives = normalize_objective(avg_objectives)
normalized_lbds = normalize(avg_lbds)
normalized_learned_clause_lengths = normalize(avg_learned_clause_lengths)
normalized_conflict_sizes = normalize(avg_conflict_sizes)

plot_runtime_and_objective(normalized_runtimes, normalized_objectives, avg_runtimes, avg_objectives)
#plot_benchmarks(normalized_runtimes, avg_runtimes, title='Normalized Benchmark Runtimes by Technique')
#plot_objective_values(normalized_objectives, avg_objectives)
#plot_benchmarks(normalized_lbds, avg_lbds, title='Normalized Benchmark LBD by Technique')
#plot_benchmarks(normalized_learned_clause_lengths, avg_learned_clause_lengths, title='Normalized Benchmark Learned Clause Length by Technique')
#plot_benchmarks(normalized_conflict_sizes, avg_conflict_sizes, title='Normalized Benchmark Conflict Size by Technique')

plot_all_statistics(normalized_runtimes, normalized_lbds, normalized_learned_clause_lengths, normalized_conflict_sizes, avg_runtimes, avg_lbds, avg_learned_clause_lengths, avg_conflict_sizes)

print_mean_ratios_and_success_rate(avg_runtimes, "Runtimes", invert=True)
print_mean_ratios_and_success_rate(avg_lbds, "LBD", remove_missing=True)
print_mean_ratios_and_success_rate(avg_learned_clause_lengths, "LCL", remove_missing=True)
print_mean_ratios_and_success_rate(avg_conflict_sizes, "CS", remove_missing=True)


#plot_all_statistics_bar(normalized_runtimes, normalized_lbds, normalized_learned_clause_lengths, normalized_conflict_sizes, avg_runtimes, avg_lbds, avg_learned_clause_lengths, avg_conflict_sizes)
# %%

directories = ["output_sudoku_single/" ]

avg_runtimes, avg_objectives, avg_lbds, avg_learned_clause_lengths, avg_conflict_sizes, std = parse_benchmark_dirs(directories)
normalized_runtimes = normalize(avg_runtimes)
normalized_objectives = normalize_objective(avg_objectives)
normalized_lbds = normalize(avg_lbds)
normalized_learned_clause_lengths = normalize(avg_learned_clause_lengths)
normalized_conflict_sizes = normalize(avg_conflict_sizes)

#print(avg_runtimes)

#plot_runtime_and_objective(normalized_runtimes, normalized_objectives, avg_runtimes, avg_objectives)
plot_benchmarks(normalized_runtimes, avg_runtimes, title='Normalized Runtimes by Technique', hide_problem_name=True)
#plot_benchmarks(avg_runtimes, avg_runtimes, title='Normalized Runtimes by Technique')

#plot_all_statistics(normalized_runtimes, normalized_lbds, normalized_learned_clause_lengths, normalized_conflict_sizes, avg_runtimes, avg_lbds, avg_learned_clause_lengths, avg_conflict_sizes)

print_mean_ratios_and_success_rate(avg_runtimes, "Runtimes", invert=True, max_runtime=MAX_RUNTIME/2)
print_mean_ratios_and_success_rate(avg_lbds, "LBD", remove_missing=True)
print_mean_ratios_and_success_rate(avg_learned_clause_lengths, "LCL", remove_missing=True)
print_mean_ratios_and_success_rate(avg_conflict_sizes, "CS", remove_missing=True)


# %%

directories = ["output_sudoku_single_new_expls_v2/" ]

avg_runtimes, avg_objectives, avg_lbds, avg_learned_clause_lengths, avg_conflict_sizes, std = parse_benchmark_dirs(directories)
normalized_runtimes = normalize(avg_runtimes)
normalized_objectives = normalize_objective(avg_objectives)
normalized_lbds = normalize(avg_lbds)
normalized_learned_clause_lengths = normalize(avg_learned_clause_lengths)
normalized_conflict_sizes = normalize(avg_conflict_sizes)

#print(avg_runtimes)

#plot_runtime_and_objective(normalized_runtimes, normalized_objectives, avg_runtimes, avg_objectives)
plot_benchmarks(normalized_runtimes, avg_runtimes, title='Normalized Runtimes by Technique', hide_problem_name=True)
#plot_benchmarks(avg_runtimes, avg_runtimes, title='Normalized Runtimes by Technique')

#plot_all_statistics(normalized_runtimes, normalized_lbds, normalized_learned_clause_lengths, normalized_conflict_sizes, avg_runtimes, avg_lbds, avg_learned_clause_lengths, avg_conflict_sizes)

print_mean_ratios_and_success_rate(avg_runtimes, "Runtimes", invert=True, max_runtime=MAX_RUNTIME/2)
print_mean_ratios_and_success_rate(avg_lbds, "LBD", remove_missing=True)
print_mean_ratios_and_success_rate(avg_learned_clause_lengths, "LCL", remove_missing=True)
print_mean_ratios_and_success_rate(avg_conflict_sizes, "CS", remove_missing=True)

# %%

directories = ["output_all_new_expls/" ]

avg_runtimes, avg_objectives, avg_lbds, avg_learned_clause_lengths, avg_conflict_sizes, std = parse_benchmark_dirs(directories)
normalized_runtimes = normalize(avg_runtimes)
normalized_objectives = normalize_objective(avg_objectives)
normalized_lbds = normalize(avg_lbds)
normalized_learned_clause_lengths = normalize(avg_learned_clause_lengths)
normalized_conflict_sizes = normalize(avg_conflict_sizes)

plot_runtime_and_objective(normalized_runtimes, normalized_objectives, avg_runtimes, avg_objectives)
plot_all_statistics(normalized_runtimes, normalized_lbds, normalized_learned_clause_lengths, normalized_conflict_sizes, avg_runtimes, avg_lbds, avg_learned_clause_lengths, avg_conflict_sizes)


# Print the mean ratios and success rates
print_mean_ratios_and_success_rate(avg_runtimes, "Runtimes", invert=True)
print_mean_ratios_and_success_rate(avg_lbds, "LBD", remove_missing=True)
print_mean_ratios_and_success_rate(avg_learned_clause_lengths, "LCL", remove_missing=True)
print_mean_ratios_and_success_rate(avg_conflict_sizes, "Conflict Size", remove_missing=True)



# Print the average runtime and standard deviation in a readable format
#print_standard_deviation(avg_runtimes, std)

# Print the overall improvement ratios
# Calculate and print runtime deviations
#calculate_runtime_deviation(avg_runtimes)

# %%
directories = ["output_all_new_expls_v2_r1/" ]

avg_runtimes, avg_objectives, avg_lbds, avg_learned_clause_lengths, avg_conflict_sizes, std = parse_benchmark_dirs(directories)
normalized_runtimes = normalize(avg_runtimes)
normalized_objectives = normalize_objective(avg_objectives)
normalized_lbds = normalize(avg_lbds)
normalized_learned_clause_lengths = normalize(avg_learned_clause_lengths)
normalized_conflict_sizes = normalize(avg_conflict_sizes)

plot_runtime_and_objective(normalized_runtimes, normalized_objectives, avg_runtimes, avg_objectives)
plot_all_statistics(normalized_runtimes, normalized_lbds, normalized_learned_clause_lengths, normalized_conflict_sizes, avg_runtimes, avg_lbds, avg_learned_clause_lengths, avg_conflict_sizes)


# Print the mean ratios and success rates
print_mean_ratios_and_success_rate(avg_runtimes, "Runtimes", invert=True)
print_mean_ratios_and_success_rate(avg_lbds, "LBD", remove_missing=True)
print_mean_ratios_and_success_rate(avg_learned_clause_lengths, "LCL", remove_missing=True)
print_mean_ratios_and_success_rate(avg_conflict_sizes, "Conflict Size", remove_missing=True)
# %%
