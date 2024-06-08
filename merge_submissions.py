#########################
## MODULES NÉCESSAIRES ##
#########################

import pandas as pd
import os
import shutil

'''
Ce fichier regroupe des fonctions utiles pour manipuler des fichiers CSV, des images, etc.
On l'utilise notamment pour fusionner des submissions faites par les modèles et celle faite
par OCR qui répond UNKNOWN si le score est inférieur au threshold (voir create_submission.py)

'''


###############
## FONCTIONS ##
###############


# Load the CSV files
df1 = pd.read_csv('/users/eleves-a/2022/hippolyte.wallaert/Modal/INF473V-challenge/submissions/submission_ocr83_2.csv')
df2 = pd.read_csv('/users/eleves-a/2022/hippolyte.wallaert/Modal/INF473V-challenge/submission_30_05_01.csv')
'''
'''
# Function to create a new merged dataframe
def merge_submissions(df1, df2):
    merged_df = df1.copy()
    for index, row in df1.iterrows():
        if row['label'] == 'UNKNOWN':
            merged_df.at[index, 'label'] = df2.at[index, 'label']
    return merged_df

# Perform the merge
merged_df = merge_submissions(df1, df2)

# Save the new merged dataframe to a new CSV file
output_path = '/users/eleves-a/2022/hippolyte.wallaert/Modal/INF473V-challenge/merged_submission_29_05_01.csv'
merged_df.to_csv(output_path, index=False)


'''
# Function to count the number of different answers between two dataframes
def count_differences(df1, df2):
    
    # Count the number of different answers
    different_count = 0
    for index, row in df1.iterrows():
        if row['label'] != df2.at[index, 'label']:
            different_count += 1
    
    return different_count

# Compute the number of different answers
different_count = count_differences(df1, df2)

# Print the result
print(f"The number of different answers between the two CSV files is: {different_count}")
'''



'''
def sort(df1):
    # On trie une copie de df1 et on crée un nouveau csv du meme nom avec sorted a la fin
    new_df = df1.copy()
    new_df = new_df.sort_values("id")
    new_df.to_csv('/users/eleves-a/2022/hippolyte.wallaert/Modal/INF473V-challenge/submission_29_05_01_sorted.csv', index=False)
    return new_df

# Perform the sort
sorted_df = sort(pd.read_csv('/users/eleves-a/2022/hippolyte.wallaert/Modal/INF473V-challenge/submission_29_05_01.csv'))
'''





'''
def move_files(src_file_path, dest_dir):
    filename = os.path.basename(src_file_path)
    new_dest_file_path = os.path.join(dest_dir, filename)
            
    dest_file_path = new_dest_file_path

        # # Move the file
        # shutil.move(src_file_path, dest_file_path)
        # print(f"Moved '{src_file_path}' to '{dest_file_path}'")

        # Copy the file
    shutil.copy(src_file_path, dest_file_path)
    print(f"Copied '{src_file_path}' to '{dest_file_path}'")

'''


'''
def count_cheese_answers(file_path):
    try:
        # Load the submission data
        submission_data = pd.read_csv(file_path)
        
        # List of cheeses to count
        cheese_list = [
            "BRIE DE MELUN", "CAMEMBERT", "EPOISSES", "FOURME D’AMBERT", "RACLETTE", "MORBIER",
            "SAINT-NECTAIRE", "POULIGNY SAINT- PIERRE", "ROQUEFORT", "COMTÉ", "CHÈVRE", "PECORINO",
            "NEUFCHATEL", "CHEDDAR", "BÛCHETTE DE CHÈVRE", "PARMESAN", "SAINT- FÉLICIEN", "MONT D’OR",
            "STILTON", "SCARMOZA", "CABECOU", "BEAUFORT", "MUNSTER", "CHABICHOU", "TOMME DE VACHE",
            "REBLOCHON", "EMMENTAL", "FETA", "OSSAU- IRATY", "MIMOLETTE", "MAROILLES", "GRUYÈRE",
            "MOTHAIS", "VACHERIN", "MOZZARELLA", "TÊTE DE MOINES", "FROMAGE FRAIS"
        ]
        
        # Initialize a dictionary to hold the counts
        cheese_counts = {cheese: 0 for cheese in cheese_list}
        
        # Count the number of answers for each cheese
        for cheese in cheese_list:
            
            cheese_counts[cheese] = submission_data['label'].str.upper().value_counts().get(cheese.upper(), 0)

        for index, row in submission_data.iterrows():
            if row['label'] == 'MONT D’OR':
                move_files(f"/users/eleves-a/2022/hippolyte.wallaert/Modal/INF473V-challenge/dataset/test/{row['id']}.jpg", f"/users/eleves-a/2022/hippolyte.wallaert/Modal/INF473V-challenge/recognized_mont")
        
        # Convert the counts to a DataFrame for better readability
        cheese_counts_df = pd.DataFrame(list(cheese_counts.items()), columns=['Cheese', 'Count'])
        
        # Print the counts
        print(cheese_counts_df)
        
        return cheese_counts_df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Provide the path to your CSV file
file_path = '/users/eleves-a/2022/hippolyte.wallaert/Modal/INF473V-challenge/submission_ocr83_2.csv'

# Call the function
cheese_counts_df = count_cheese_answers(file_path)
'''


        

