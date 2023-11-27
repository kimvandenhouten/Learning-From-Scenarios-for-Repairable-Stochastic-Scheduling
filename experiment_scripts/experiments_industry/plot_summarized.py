import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

""" 
This Python script plots the normalized post-hoc-regret loss in a summarizing boxplot comparing decision-focused, 
stochastic and deterministic. The plots can be found in plots/final_evaluation/normalized...
"""

problem_type = "RCPSP_penalty"

for penalty_type in ["small", "large"]:
    for instance_set in ["industry_small", "industry_large"]:
        combined_data = pd.DataFrame()
        for i, folder_path in enumerate([f"results/{instance_set}/{penalty_type}", f"results/{instance_set}/{penalty_type}"]):
            # Loop through CSV files in the folder
            for filename in os.listdir(folder_path):
                if filename.endswith(".csv"):
                    file_path = os.path.join(folder_path, filename)
                    print(file_path)
                    df = pd.read_csv(file_path)
                    # Append the DataFrame to the combined_data DataFrame
                    combined_data = pd.concat([combined_data, df], ignore_index=True)

        combined_data["instance"] = combined_data["instance"].str.replace("_factory_1", "")
       
        for col_name in ["total_time"]:
            combined_data[col_name] = combined_data[col_name].round(2)

        # Calculate the maximum test loss for each instance
        max_test_loss = combined_data.groupby("instance")["test_loss"].max()

        # Merge the maximum test loss back into the combined_data DataFrame
        combined_data = pd.merge(combined_data, max_test_loss, on="instance", suffixes=("", "_max"))

        # Create a new column for normalized test loss
        combined_data["normalized_test_loss"] = combined_data["test_loss"] / combined_data["test_loss_max"]

        fontsize = 32
        df = combined_data
        legend_order = ["decision-focused", "deterministic", "stochastic"]
        sns.set_palette("colorblind")

        plt.figure(figsize=(4, 4))

        bar = sns.boxplot(data=df, x="algorithm", y="normalized_test_loss", order=legend_order)
        plt.xticks(rotation=45, fontsize=30, label=False)
        bar.set(xlabel=None)
        bar.set(xticklabels=[])
        plt.yticks([0, 0.5, 1], fontsize=30)  # Set y-axis label fontsize on seaborn axis

        if True:
            hatches = ['/', 'o', '-']
            for i, thisbar in enumerate(bar.patches):
                thisbar.set_hatch(hatches[i])

        plt.tight_layout(rect=[0.05, 0, 0.95, 0.9])
        label = "Norm. p-h regret"
        plt.ylabel(label, fontsize=fontsize)
        if penalty_type == "small":
            title = "Post-hoc regret, rho=1/size"
        elif penalty_type == "large":
            title = "Post-hoc regret, rho=1"
        elif penalty_type == "medium":
            title = "Post-hoc regret, rho=1/products"
        title = f"Penalty {penalty_type}"
        plt.title(title, fontsize=fontsize)
        plt.savefig(f'plots/final_evaluation/summarized_{instance_set}_penalty={penalty_type}.png')
        plt.close()