import pandas as pd
import os

#Header
#Gaze 3D position left X, Gaze 3D position left Y, Gaze 3D position left Z, Gaze 3D position right X,Gaze 3D position right Y,Gaze 3D position right Z,Pupil diameter left,Pupil diameter right, HR, GSR, ConationLevel

#Path to data folder
path = r"C:\Users\dines\Desktop\Projects\ML-Conation\ConationNetwork\Data\\"

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

    df = df[~(df == 0.0).any(axis=1)]

    baselinelist = df.iloc[1:1000, [8]].mean(axis=0)
    df.HR = df[df.columns[[8]]] - baselinelist

    df = df.drop(df.columns[[9, 10]], axis=1)


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


dfnew = pd.read_csv(r'C:\Users\dines\Desktop\Projects\ML-Conation\ConationNetwork\Data\\CombinedData.csv', header=0, sep=',')
dfold = pd.read_csv(r'C:\Users\dines\Desktop\Projects\ML-Conation\ConationNetwork\Data\\CombinedDataold.csv', header=0, sep=',')

dfnew[dfnew.columns[[0,1,2,3,4,5]]] = dfold[dfold.columns[[0,1,2,3,4,5]]]

dfnew.to_csv('CombinedDataOldNew.csv', index=False)
print('Done')


#1:len(big_df)-1