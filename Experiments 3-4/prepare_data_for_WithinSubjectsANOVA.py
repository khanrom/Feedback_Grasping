import pandas as pd
import json

file_names = ['test_losses_DIFF_conn_1_2PC_spec_targ_fixed_dist_PC2_fb_even_longer_bb.json', 'test_losses_DIFF_conn_2_2PC_spec_targ_fixed_dist_PC2_fb_even_longer_bb.json', 'test_losses_DIFF_conn_3_2PC_spec_targ_fixed_dist_PC2_fb_even_longer_bb.json'] #['rev_test_losses_conn_1_2PC.json', 'rev_test_losses_conn_2_2PC.json', 'rev_test_losses_conn_3_2PC.json']
feedback_types = ['1', '2', '3']

# Initialize an empty DataFrame
columns = ['object_id']# + [f'({ft}, {nl})' for ft in feedback_types for nl in range(1, 6)]
data = {col: [] for col in columns}
df = pd.DataFrame(data)
df['object_id'] = range(1, 9383)  # Generate object IDs from 1 to 9382

# Load data from each file and populate the DataFrame
for file_name, feedback_type in zip(file_names, feedback_types):
    with open(file_name, 'r') as file:
        json_data = json.load(file)
        for noise_level, losses in json_data.items():
            column_name = f'({feedback_type}, {noise_level})'
            df[column_name] = losses  # Add test losses to the dataframe

# Save the dataframe to a CSV file
df.to_csv('test_losses_DIFF_Exp_1ciii.csv', index=False)
