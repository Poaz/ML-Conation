import pandas as pd
import os

#Path to data folder
path = r"C:\Users\dines\Desktop\Projects\ML-Conation\ConationNetwork\Data5\\"

# List to hold file names
FileNames = []

#set path
os.chdir(r'' + path)

# Find any file that ends with ".txt"
for files in os.listdir("."):
    if files.endswith(".txt"):
        FileNames.append(files)

def GetFile(fnombre):
    location = r'' + path + fnombre
    # Parse the txt file
    df = pd.read_csv(location, header=0, sep=',')

    # Tag record to file name
    df['File'] = fnombre
    #df = df
    #df = df.drop(['GameState'], axis=1)
    #df = df.drop(['TimeSinceStart'], axis=1)
    # Make the "File" column the index of the df
    return df.set_index(['File'])

# Create a list of dataframes
df_list = [GetFile(fname) for fname in FileNames]

# Combine all of the dataframes into one
big_df = df_list[0]
for i in range(len(df_list)):
        big_df = big_df.append(df_list[i])


#concated = pd.read_csv(r"C:\Users\dines\Desktop\Projects\ML-Conation\ConationNetwork\CombinedData_Data3.csv", header=0, sep=',')
#concated['File'] = 'CombinedData_Data3.csv'
#concated.set_index(['File'])

#big_df = big_df.append(concated)
big_df.fillna(0)
big_df.to_csv('CombinedData.csv', index=False)

