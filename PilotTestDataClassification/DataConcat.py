import pandas as pd
import os
import sys

# List to hold file names
FileNames = []

# Your path will be different, please modify the path below.
os.chdir(r"C:\Users\dines\Google Drev\School\AAU\8. Semester\Semester Project 8\PilotTestDataClassification\Data")

# Find any file that ends with ".csv"
for files in os.listdir("."):
    if files.endswith(".csv"):
        FileNames.append(files)

def GetFile(fnombre):
    # Path to excel file
    # Your path will be different, please modify the path below.
    location = r'C:\Users\dines\Google Drev\School\AAU\8. Semester\Semester Project 8\PilotTestDataClassification\Data\\' + fnombre

    # Parse the excel file
    # 0 = first sheet
    df = pd.read_csv(location, header=0, sep=';')

    # Tag record to file name
    df['File'] = fnombre

    # Make the "File" column the index of the df
    return df.set_index(['File'])


def GetFileComma(fnombre):
    # Path to excel file
    # Your path will be different, please modify the path below.
    location = r'C:\Users\dines\Google Drev\School\AAU\8. Semester\Semester Project 8\PilotTestDataClassification\Data\\' + fnombre

    # Parse the excel file
    # 0 = first sheet
    df = pd.read_csv(location, header=0, sep=',')

    # Tag record to file name
    df['File'] = fnombre

    # Make the "File" column the index of the df
    return df.set_index(['File'])

# Create a list of dataframes
df_list = [GetFile(fname) for fname in FileNames]

eyetrack = GetFile("P4.csv")
naos = GetFileComma("P4_naos.csv")
eyetrack.reset_index(drop=True)
naos.reset_index(drop=True)


# Combine all of the dataframes into one
big_df = pd.concat([eyetrack, naos], axis=1, join_axes=[eyetrack.index])
#print(big_df)

print(eyetrack)
print(naos)
#combined_df = eyetrack.join(naos, lsuffix='eyetrack', rsuffix='naos')

big_df.fillna(0)
big_df.to_csv('CombinedData.csv', index=False)

print('Done')