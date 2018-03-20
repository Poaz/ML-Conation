import pandas as pd
import os
import sys
import numpy as np

# List to hold file names
FileNames = []

# Your path will be different, please modify the path below.
os.chdir(r"C:\Users\dines\Google Drev\School\AAU\8. Semester\Semester Project 8\Machine Learning\ML-Conation\PilotTestDataClassification\Data\\")

# Find any file that ends with ".csv"
for files in os.listdir("."):
    if files.endswith(".txt"):
        FileNames.append(files)

def GetFile(fnombre):
    # Path to excel file
    # Your path will be different, please modify the path below.
    location = r'C:\\Users\\dines\\Google Drev\\School\\AAU\\8. Semester\\Semester Project 8\\Machine Learning\\ML-Conation\\PilotTestDataClassification\\Data\\' + fnombre

    # Parse the excel file
    # 0 = first sheet
    df = pd.read_csv(location, header=0, sep=',')

    # Tag record to file name
    df['File'] = fnombre

    # Make the "File" column the index of the df
    return df.set_index(['File'])

# Create a list of dataframes
#df_list = [GetFile(fname) for fname in FileNames]

df_list = [GetFile('TrimmedOutput_Participant_10.txt'), GetFile('TrimmedOutput_Participant_11.txt'), GetFile('TrimmedOutput_Participant_1.txt')]

# Combine all of the dataframes into one
#big_df = pd.concat(df_list, axis=1, ignore_index=True)
big_df = df_list[0].append(df_list[1])
big_df = big_df.append(df_list[2])
big_df.fillna(0)
#big_df = big_df.iloc[:, 12].astype(int)
big_df.to_csv('CombinedData.csv', index=False)
print('Done')