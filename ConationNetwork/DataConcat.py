import pandas as pd
import os

#Header
#Gaze 3D position left X, Gaze 3D position left Y, Gaze 3D position left Z, Gaze 3D position right X,Gaze 3D position right Y,Gaze 3D position right Z,Pupil diameter left,Pupil diameter right, HR, GSR, ConationLevel

#Path to data folder
path = r"D:\Google Drive\School\AAU\8. Semester\Semester Project 8\Machine Learning\ML-Conation\ConationNetwork\Data\\"

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

    # Make the "File" column the index of the df
    return df.set_index(['File'])

# Create a list of dataframes
df_list = [GetFile(fname) for fname in FileNames]

# Combine all of the dataframes into one
big_df = df_list[0]
for i in  range(len(df_list)):
        big_df = big_df.append(df_list[i])

big_df.fillna(0)
#big_df = big_df.apply(pd.to_numeric)
#print(len(big_df)-1)
#big_df = big_df.iloc[:, 12].astype(int)
big_df.to_csv('CombinedData.csv', index=False)
print('Done')


#1:len(big_df)-1