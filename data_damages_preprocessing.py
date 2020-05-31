import os
import urllib.request
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import warnings
warnings.filterwarnings("ignore")

'''
Get the ID of the car listed
'''
def get_id(row):
    url = row['listing_url']
    listing_id = url[-8:]
    return listing_id

'''
Save the four images associated with each car listing
'''
def save_images(row, data_path):
    listing_id = row['listing_id']
    image1_url = row['image1_url']
    image2_url = row['image2_url']
    image3_url = row['image3_url']
    image4_url = row['image4_url']
    
    image_urls = [image1_url, image2_url, image3_url, image4_url]
    for i, url in enumerate(image_urls):
        urllib.request.urlretrieve(url, data_path + '{}-{}.jpg'.format(listing_id, i+1))
        
def main():
    print("Fetching files...")
    car_damages_path = 'car_damages.csv'
    car_damages_cleaned_path = 'car_damages_cleaned.csv'

    car_damages = pd.read_csv(car_damages_path)
    car_damages.columns = ['listing_url', 'year', 'make', 'model', 'damage', 'est_value', 'label', 'image1_url', 'image2_url', 'image3_url', 'image4_url']
    
    print("Cleaning data...")
    car_damages['listing_id'] = car_damages.apply(lambda row: get_id(row), axis=1)
    car_damages = car_damages[['listing_id','listing_url', 'year', 'make', 'model', 'damage', 'est_value', 'label', 'image1_url', 'image2_url', 'image3_url', 'image4_url']]
    car_damages = car_damages.dropna()
    
    print("Saving images to local machine in ./data_damages...")
    car_damages.to_csv(car_damages_cleaned_path, index=False)
    data_path = './data_damages/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    else:
        shutil.rmtree(data_path)           
        os.makedirs(data_path)
    car_damages.apply(lambda row: save_images(row, data_path), axis=1)
    
    print("Finished!")
    
    
if __name__ == "__main__":
    main()