import pandas as pd

# Load the CSV files
df1 = pd.read_csv('/users/eleves-a/2022/hippolyte.wallaert/Modal/INF473V-challenge/submission_ocr_83.csv')
df2 = pd.read_csv('/users/eleves-a/2022/hippolyte.wallaert/Modal/INF473V-challenge/submissions/submission28_05_1.csv')

# Function to merge the two dataframes based on the specified logic
def merge_submissions(df1, df2):
    merged_df = df1.copy()
    for index, row in merged_df.iterrows():
        if row['label'] == 'UNKNOWN':
            answer = df2[df2['id'] == row['id']]['label'].values
            if len(answer) > 0:
                merged_df.at[index, 'label'] = answer[0]
    return merged_df

# Perform the merge
merged_df = merge_submissions(df1, df2)

# Save the merged dataframe to a new CSV file
merged_df.to_csv('merged_submission.csv', index=False)

