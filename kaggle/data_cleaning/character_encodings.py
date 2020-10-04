import pandas as pd
import numpy as np
# helpful character encoding module
import chardet

# # 1) What are encodings?
# Character encodings are specific sets of rules for mapping from raw binary byte strings
# UTF-8 is the standard text encoding. All Python code is in UTF-8 and, ideally, all your data should be as well. It's when things aren't in UTF-8 that you run into trouble.

sample_entry = b'\xa7A\xa6n' # you're working with a dataset composed of bytes.  
# create a variable `new_entry` that changes the encoding from `"big5-tw"` to `"utf-8"`.  `new_entry` should have the bytes datatype.
before = sample_entry.decode("big5-tw")
new_entry = before.encode()

# # 2) Reading in files with encoding problems
# Figure out what the correct encoding should be and read in the file to a DataFrame `police_killings`.

with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))

print(result) # returns {'encoding': 'Windows-1252', 'confidence': 0.73, 'language': ''}

police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='Windows-1252')

# # 3) Saving your files with UTF-8 encoding
# Save a version of the police killings dataset to CSV with UTF-8 encoding.

police_killings.to_csv("my_file.csv")