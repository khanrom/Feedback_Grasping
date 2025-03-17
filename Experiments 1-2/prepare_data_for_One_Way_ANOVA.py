import json
import pandas as pd

# Load the JSON data
with open('test_losses_DIFF_conn_2_2PC.json', 'r') as file:
    data = json.load(file)

# Convert the JSON data into a long-format DataFrame
long_format = pd.DataFrame(
    [(key, value) for key, values in data.items() for value in values],
    columns=['Group', 'Value']
)

# Save the long-format DataFrame to a CSV file
output_path = 'test_losses_DIFF_conn_2_2PC.csv'
long_format.to_csv(output_path, index=False)

print(f"Data successfully converted to long format and saved to {output_path}")
