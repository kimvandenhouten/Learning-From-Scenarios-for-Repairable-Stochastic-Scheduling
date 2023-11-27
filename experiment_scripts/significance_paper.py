import os
import pandas as pd
from scipy.stats import ttest_rel

# settings
for directory in ["j30", "j90", "industry_small", "industry_large"]:
    for penalty_type in ["small", "large"]:
        for (Algorithm1, Algorithm2) in [("deterministic", "stochastic"), ("deterministic", "decision-focused"),
                                         ("stochastic", "decision-focused")]:
            folder_path = f"results/{directory}/{penalty_type}"
            # Loop through CSV files in the folder
            combined_data = pd.DataFrame()
            for filename in os.listdir(folder_path):
                if filename.endswith(".csv"):
                    file_path = os.path.join(folder_path, filename)
                    df = pd.read_csv(file_path)

                    # Append the DataFrame to the combined_data DataFrame
                    combined_data = pd.concat([combined_data, df], ignore_index=True)

            for col_name in ["total_time"]:
                combined_data[col_name] = combined_data[col_name].round(2)

            df = combined_data
            df = df.sort_values(by='instance')

            algorithm1_data = df[df['algorithm'] == Algorithm1]['test_loss']
            algorithm2_data = df[df['algorithm'] == Algorithm2]['test_loss']

            # Perform t-test
            alpha = 0.05
            t_statistic, p_value = ttest_rel(algorithm1_data, algorithm2_data)

            # Print the results
            print(f'For {directory} and penalty type {penalty_type} and comparing {Algorithm1} and {Algorithm2} with alpha {alpha}')
            print(f'T-statistic: {t_statistic}')
            print(f'P-value: {p_value}')

            # Check for significance based on the p-value
            if p_value < alpha:
                print('The difference is statistically significant.')

                if t_statistic < 0:
                    print(f'{Algorithm1} has a statistically significantly lower test_loss (better performance) than {Algorithm2}.')
                else:
                    print(f'{Algorithm2} has a statistically significantly lower test_loss (better performance) than {Algorithm1}.')
            else:
                print('The difference is not statistically significant. No conclusive evidence.')

            print('\n')