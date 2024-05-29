import pandas as pd

# Load the CSV files
df1 = pd.read_csv('/users/eleves-a/2022/hippolyte.wallaert/Modal/INF473V-challenge/submission_ocr_83.csv')
df2 = pd.read_csv('/users/eleves-a/2022/hippolyte.wallaert/Modal/INF473V-challenge/submissions/submission28_05_1.csv')

# Function to create a new merged dataframe
def merge_submissions(df1, df2):
    merged_df = df2.copy()
    for index, row in df2.iterrows():
        if row['label'] == 'UNKNOWN':
            match = df1[df1['id'] == row['id']]
            if not match.empty:
                merged_df.at[index, 'label'] = match.iloc[0]['label']
    return merged_df

# Perform the merge
merged_df = merge_submissions(df1, df2)

# Save the new merged dataframe to a new CSV file
output_path = '/users/eleves-a/2022/hippolyte.wallaert/Modal/INF473V-challenge/merged_submission.csv'
merged_df.to_csv(output_path, index=False)

# Display the new merged dataframe for debugging
print("First few rows of the merged dataframe:")
print(merged_df.head())

# Function to count the number of different answers between two dataframes
def count_differences(df1, df2):
    # Merge the dataframes on 'id'
    merged_df = pd.merge(df1, df2, on='id', suffixes=('_original', '_merged'))
    
    # Count the number of different answers
    different_count = (merged_df['label_original'] != merged_df['label_merged']).sum()
    
    return different_count

# Compute the number of different answers
different_count = count_differences(df2, merged_df)

# Print the result
print(f"The number of different answers between the two CSV files is: {different_count}")
