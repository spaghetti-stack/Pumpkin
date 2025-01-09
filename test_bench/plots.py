# %%
import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

#%matplotlib qtagg

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
        for (problem, data_file), techniques in learned_clause_length_data.items()
    }
    avg_conflict_sizes = {
        (problem, data_file): {tech: sum(sizes) / len(sizes) if sizes else 0 for tech, sizes in techniques.items()}
        for (problem, data_file), techniques in conflict_size_data.items()
    }
    return (avg_runtimes, avg_objectives, avg_lbds, avg_learned_clause_lengths, avg_conflict_sizes)

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
            tech: (value / decomp_value if decomp_value > 0 else value) for tech, value in tech_values.items()
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

def plot_benchmarks(avg_values, abs_values, title='Normalized Benchmark Values by Technique'):
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

    # Plotting
    problems = list(grouped_data.keys())
    abbreviated_problems = [f"{abbreviate_text(problem, max_length=10)}, {abbreviate_text(data_file, max_length=20)}" for problem, data_file in problems]
    x = range(len(problems))
    bar_width = 0.2

    plt.figure(figsize=(12, 6))

    # Draw bars and annotate values
    for i, technique in enumerate(techniques):
        values = [group[i] for group in grouped_data.values()]
        values_abs = [group[i] for group in grouped_data_abs.values()]
        bar_positions = [pos + i * bar_width for pos in x]
        plt.bar(
            bar_positions,
            values,
            bar_width,
            label=technique,
            color=colors[technique]
        )
        # Annotate bars with values
        for pos, value in zip(bar_positions, values):
            plt.text(pos, value + 0.05, f"{value:.2f}", ha='center', va='bottom', fontsize=6)

        if technique == "decomp":
            for pos, value in zip(bar_positions, values_abs):
                plt.text(pos, -0.3, f"{value:.1f}", ha='center', va='bottom', fontsize=8)

    # Draw a baseline for "decomp"
    decomp_positions = [pos + bar_width * 0 for pos in x]  # First position corresponds to "decomp"
    plt.plot(decomp_positions, [1] * len(decomp_positions), '-', label='Baseline (Decomp = 1)', color="black", lw=0.5)

    plt.xlabel('Problems (Problem, Data File)')
    plt.ylabel('Normalized Value (Relative to Decomp)')
    plt.title(title)
    plt.xticks([pos + bar_width for pos in x], abbreviated_problems, rotation=45, ha='right')
    plt.legend(title="Technique")
    plt.tight_layout()
    plt.show()

def plot_all_statistics(avg_runtimes, avg_lbds, avg_learned_clause_lengths, avg_conflict_sizes, abs_runtimes, abs_lbds, abs_learned_clause_lengths, abs_conflict_sizes):
    """
    Creates scatter plots of benchmark statistics (runtime, LBD, learned clause length, and conflict size) by technique.

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

    problems = list(grouped_data_runtime.keys())
    abbreviated_problems = [f"{abbreviate_text(problem, max_length=10)}, {abbreviate_text(data_file, max_length=20)}" for problem, data_file in problems]
    x = range(len(problems))
    bar_width = 0.2

    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1]})

    # Plot runtime scatter plot
    for i, technique in enumerate(techniques):
        runtimes = [group[i] for group in grouped_data_runtime.values()]
        runtimes_abs = [group[i] for group in grouped_data_abs_runtime.values()]
        x_positions = [pos + i * bar_width for pos in x]
        axs[0].scatter(
            x_positions,
            runtimes,
            label=f'{technique} (Runtime)',
            color=colors[technique],
            marker=markers[technique],
            edgecolor="black",
            s=100  # Marker size
        )
        # Annotate points with runtimes
        for pos, runtime, abs_runtime in zip(x_positions, runtimes, runtimes_abs):
            axs[0].text(pos, runtime + 0.05, f"{runtime:.2f}", ha='center', va='bottom', fontsize=6)
            if technique == "decomp":  # Additional annotation for "decomp"
                axs[0].text(pos, runtime - 0.3, f"{abs_runtime:.1f}", ha='center', va='top', fontsize=8)

    # Draw a baseline for "decomp"
    decomp_positions = [pos + bar_width * 0 for pos in x]  # First position corresponds to "decomp"
    axs[0].plot(decomp_positions, [1] * len(decomp_positions), '-', label='Baseline (Decomp = 1)', color="black", lw=0.5)

    axs[0].set_ylabel('Normalized Runtime (Relative to Decomp)')
    axs[0].set_title('Normalized Benchmark Runtimes by Technique')
    axs[0].legend(title="Technique")

    # Plot LBD scatter plot
    for i, technique in enumerate(techniques):
        lbds = [group[i] for group in grouped_data_lbd.values()]
        lbds_abs = [group[i] for group in grouped_data_abs_lbd.values()]
        x_positions = [pos + i * bar_width for pos in x]
        axs[1].scatter(
            x_positions,
            lbds,
            label=f'{technique} (LBD)',
            color=colors[technique],
            marker=markers[technique],
            edgecolor="black",
            s=100,  # Marker size
            alpha=0.5  # Transparency for LBD
        )
        # Annotate points with LBDs
        for pos, lbd, abs_lbd in zip(x_positions, lbds, lbds_abs):
            axs[1].text(pos, lbd + 0.05, f"{lbd:.2f}", ha='center', va='bottom', fontsize=6, alpha=0.5)
            if technique == "decomp":  # Additional annotation for "decomp"
                axs[1].text(pos, lbd - 0.3, f"{abs_lbd:.1f}", ha='center', va='top', fontsize=8, alpha=0.5)

    # Draw a baseline for "decomp" on LBD axis
    axs[1].plot(decomp_positions, [1] * len(decomp_positions), '-', label='Baseline (Decomp = 1)', color="black", lw=0.5)

    axs[1].set_ylabel('Normalized LBD (Relative to Decomp)')
    axs[1].set_title('Normalized Benchmark LBD by Technique')
    axs[1].legend(title="Technique")

    # Plot learned clause length scatter plot
    for i, technique in enumerate(techniques):
        learned_clause_lengths = [group[i] for group in grouped_data_learned_clause_length.values()]
        learned_clause_lengths_abs = [group[i] for group in grouped_data_abs_learned_clause_length.values()]
        x_positions = [pos + i * bar_width for pos in x]
        axs[2].scatter(
            x_positions,
            learned_clause_lengths,
            label=f'{technique} (Learned Clause Length)',
            color=colors[technique],
            marker=markers[technique],
            edgecolor="black",
            s=100,  # Marker size
            alpha=0.5  # Transparency for learned clause length
        )
        # Annotate points with learned clause lengths
        for pos, length, abs_length in zip(x_positions, learned_clause_lengths, learned_clause_lengths_abs):
            axs[2].text(pos, length + 0.05, f"{length:.2f}", ha='center', va='bottom', fontsize=6, alpha=0.5)
            if technique == "decomp":  # Additional annotation for "decomp"
                axs[2].text(pos, length - 0.3, f"{abs_length:.1f}", ha='center', va='top', fontsize=8, alpha=0.5)

    # Draw a baseline for "decomp" on learned clause length axis
    axs[2].plot(decomp_positions, [1] * len(decomp_positions), '-', label='Baseline (Decomp = 1)', color="black", lw=0.5)

    axs[2].set_ylabel('Normalized Learned Clause Length (Relative to Decomp)')
    axs[2].set_title('Normalized Benchmark Learned Clause Length by Technique')
    axs[2].legend(title="Technique")

    # Plot conflict size scatter plot
    for i, technique in enumerate(techniques):
        conflict_sizes = [group[i] for group in grouped_data_conflict_size.values()]
        conflict_sizes_abs = [group[i] for group in grouped_data_abs_conflict_size.values()]
        x_positions = [pos + i * bar_width for pos in x]
        axs[3].scatter(
            x_positions,
            conflict_sizes,
            label=f'{technique} (Conflict Size)',
            color=colors[technique],
            marker=markers[technique],
            edgecolor="black",
            s=100,  # Marker size
            alpha=0.5  # Transparency for conflict size
        )
        # Annotate points with conflict sizes
        for pos, size, abs_size in zip(x_positions, conflict_sizes, conflict_sizes_abs):
            axs[3].text(pos, size + 0.05, f"{size:.2f}", ha='center', va='bottom', fontsize=6, alpha=0.5)
            if technique == "decomp":  # Additional annotation for "decomp"
                axs[3].text(pos, size - 0.3, f"{abs_size:.1f}", ha='center', va='top', fontsize=8, alpha=0.5)

    # Draw a baseline for "decomp" on conflict size axis
    axs[3].plot(decomp_positions, [1] * len(decomp_positions), '-', label='Baseline (Decomp = 1)', color="black", lw=0.5)

    axs[3].set_xlabel('Problems (Problem, Data File)')
    axs[3].set_ylabel('Normalized Conflict Size (Relative to Decomp)')
    axs[3].set_title('Normalized Benchmark Conflict Size by Technique')
    axs[3].legend(title="Technique")

    axs[0].set_xticks([pos + bar_width for pos in x])
    axs[3].set_xticklabels(abbreviated_problems, rotation=45, ha='right')

    fig.tight_layout()
    plt.show()

# %%
# Replace with your list of directories containing benchmark files
#directories = ["output_evm_super_compilation/", "output_community_detection/", "output_physician_scheduling/", "output_rotating_workforce_scheduling/", "output_vaccine/"]
directories = ["output_evm_super_compilation/", "output_community_detection/", "output_rotating_workforce_scheduling/", "output_community_detection_rnd/" ]

avg_runtimes, avg_objectives, avg_lbds, avg_learned_clause_lengths, avg_conflict_sizes = parse_benchmark_dirs(directories)
normalized_runtimes = normalize(avg_runtimes)
normalized_lbds = normalize(avg_lbds)
normalized_learned_clause_lengths = normalize(avg_learned_clause_lengths)
normalized_conflict_sizes = normalize(avg_conflict_sizes)

plot_benchmarks(normalized_runtimes, avg_runtimes, title='Normalized Benchmark Runtimes by Technique')
plot_benchmarks(normalized_lbds, avg_lbds, title='Normalized Benchmark LBD by Technique')
plot_benchmarks(normalized_learned_clause_lengths, avg_learned_clause_lengths, title='Normalized Benchmark Learned Clause Length by Technique')
plot_benchmarks(normalized_conflict_sizes, avg_conflict_sizes, title='Normalized Benchmark Conflict Size by Technique')

plot_all_statistics(normalized_runtimes, normalized_lbds, normalized_learned_clause_lengths, normalized_conflict_sizes, avg_runtimes, avg_lbds, avg_learned_clause_lengths, avg_conflict_sizes)

# %%
