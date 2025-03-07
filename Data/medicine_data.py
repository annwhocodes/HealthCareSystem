import zipfile
import os

zip_file_path = "C:/Users/Ananya/Desktop/Hackathon_Project/Data/archive.zip" 
extract_to = "C:/Users/Ananya/Desktop/Hackathon_Project/Data/"

if os.path.exists(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction successful!")
else:
    print("ZIP file not found! Check the file name and path.")
