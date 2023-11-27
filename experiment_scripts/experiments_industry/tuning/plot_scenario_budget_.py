import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # For color palette
import os
combined_data = pd.DataFrame()

penalty_type = "large"
algorithms = ["decision-focused", "deterministic", "stochastic"]
directory = f"results/tuning/industry_tuning/{penalty_type}"
for i, folder_path in enumerate([f"{directory}/sfge_scenarios", f"{directory}/deterministic_scenarios", f"{directory}/stochastic_scenarios"]):
    # Loop through CSV files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            print(file_path)
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            df["algorithm"] = algorithms[i]

            # Append the DataFrame to the combined_data DataFrame
            combined_data = combined_data.append(df, ignore_index=True)

combined_data["instance"] = combined_data["instance"].str.replace("_factory_1", "")

df = combined_data
custom_order = [5, 10, 25, 50, 100]
df['scenarios'] = pd.Categorical(df['scenarios'], categories=custom_order, ordered=True)
plt.rcParams.update({'font.size': 20})

# Create a color palette with a unique color for each algorithm
unique_algorithms = algorithms
num_algorithms = len(unique_algorithms)
palette = sns.color_palette("colorblind")
markers = {"decision-focused": '.', "deterministic": 'X', "stochastic": 'P'}
# Create a dictionary to map algorithms to colors
algorithm_colors = dict(zip(unique_algorithms, palette))

# Group the DataFrame by 'instance_name'
grouped_instances = df.groupby('instance')

# Create a separate figure for the legend
legend_fig, legend_ax = plt.subplots(figsize=(6, 3))  # Adjust the size as needed

# Initialize an empty legend
legend_handles = []

# Plot each instance separately in different figures
for instance_name, instance_group in grouped_instances:

    plt.figure(figsize=(7, 5))  # Create a new figure for each instance

    # Group the instance data by 'algorithm'
    grouped_algorithms = instance_group.groupby('algorithm')

    for algorithm, algorithm_group in grouped_algorithms:
        algorithm_group = algorithm_group.sort_values(by='scenarios')
        color = algorithm_colors.get(algorithm, 'b')  # Default to blue if not found in the palette

        plt.plot(algorithm_group['scenarios'], algorithm_group['training_loss'],
                 label=f"{algorithm} training loss", color=color, linestyle='-', marker=markers[algorithm], markersize=10)
        plt.plot(algorithm_group['scenarios'], algorithm_group['val_loss'],
                 label=f"{algorithm} validation loss", color=color, linestyle='--', marker=markers[algorithm], markersize=10)

    # Add labels and a title for the instance
    plt.xlabel('Number of scenarios')
    plt.ylabel('Post-hoc regret loss')
    plt.title(f'Training and validation loss for {instance_name}')
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    unique_budgets = instance_group['scenarios'].unique()
    plt.xticks([10, 25, 50, 100])
    plt.savefig(f'plots/industry_tuning/scenario_budget_instance_{instance_name}')

    # Capture the handles and labels for the legend
    handles, labels = plt.gca().get_legend_handles_labels()

    # Append the handles and labels to the legend_handles list
    legend_handles.extend(handles)

    # Add the legend to the separate legend figure
    legend_ax.legend(legend_handles, labels, loc="center")
    legend_ax.axis('off')  # Hide the axis for the legend figure

    # Save the legend as a separate figure
    legend_fig.savefig(f'plots/industry_tuning/scenario_budget_legend.png')