#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import sys


# In[26]:


if len(sys.argv) != 2:
    print("Usage: python3 load.py <file_path>")
    sys.exit(1)
file_path = sys.argv[1]


# In[2]:


df = pd.read_csv("Housing.csv")


# In[3]:


df


# In[4]:


df.info()


# In[7]:


df.describe()


# In[12]:


missing_values_summary = df.isnull().sum()

print("Missing Values Summary:")
print(missing_values_summary)


# ## **Property with the highest number of rooms and parking and the highest price:**

# In[14]:


# Sort the DataFrame by 'bedrooms', 'parking', and 'price' columns in descending order
sorted_data = df.sort_values(by=['bedrooms', 'parking', 'price'], ascending=[False, False, False])

property_with_highest_rooms_and_parking = sorted_data.iloc[0]


print("Property with the highest number of rooms and parking and the highest price:")
print(property_with_highest_rooms_and_parking)


# ## **Price analysis for furnishingstatu**

# In[18]:


# Calculate the average price for each furnishing status category

average_price_furnished = df[df['furnishingstatus'] == 'furnished']['price'].mean()
average_price_semi_furnished = df[df['furnishingstatus'] == 'semi-furnished']['price'].mean()
average_price_unfurnished = df[df['furnishingstatus'] == 'unfurnished']['price'].mean()


print("Average Price for Furnished Properties:", average_price_furnished)
print("Average Price for Semi-Furnished Properties:", average_price_semi_furnished)
print("Average Price for Unfurnished Properties:", average_price_unfurnished)

average_prices = {
    'Furnished': average_price_furnished,
    'Semi-Furnished': average_price_semi_furnished,
    'Unfurnished': average_price_unfurnished
}

max_category_name, max_average_price = max(average_prices.items(), key=lambda x: x[1])



print("\nCategory with Maximum Average Price:", max_category_name)
print("Maximum Average Price:", max_average_price)


# ## Property with the Biggest Area on Main Road and Average Price of Main Road Properties

# In[22]:


# Filter the DataFrame to select properties located on a main road
main_road_properties = df[df['mainroad'] == 'yes']

# Find the property with the biggest area among main road properties
biggest_area_property = main_road_properties[main_road_properties['area'] == main_road_properties['area'].max()]

# Calculate the average price of main road properties
average_price_main_road = main_road_properties['price'].mean()


print("Property with the Biggest Area on Main Road:")
print(biggest_area_property)

print("Average Price of Main Road Properties:", average_price_main_road)

#Max house prices
print("Max house prices", df["price"].max())


# In[23]:


# Group the data by 'prefarea' and 'furnishingstatus'
average_price_by_area_and_furnishing = df.groupby(['prefarea', 'furnishingstatus'])['price'].mean()

#  the average prices by preferred area and furnishing
print("Average Prices by Preferred Area and Furnishing Status:")
print(average_price_by_area_and_furnishing)


# In[28]:


#file_path = "eda-in-1.txt"

# Open the file in write mode and write the insights to it
with open(file_path, "w") as file:
    file.write("Insights and Results:\n\n")

    # Insight 1
    file.write("Insight 1: Property with the highest number of rooms and parking and the highest price:\n")
    file.write(property_with_highest_rooms_and_parking.to_string() + "\n\n")

    # Insight 2
    file.write("Insight 2: Price analysis for furnishing status\n")
    file.write("Average Price for Furnished Properties:\n")
    file.write(str(average_price_furnished) + "\n\n")
    file.write("Average Price for Semi-Furnished Properties:\n")
    file.write(str(average_price_semi_furnished) + "\n\n")
    file.write("Average Price for Unfurnished Properties:\n")
    file.write(str(average_price_unfurnished) + "\n\n")

    # Insight 3 (Property on Main Road)
    file.write("Insight 3: Property with the Biggest Area on Main Road:\n")
    file.write(biggest_area_property.to_string() + "\n\n")
    file.write("Average Price of Main Road Properties:\n")
    file.write(str(average_price_main_road) + "\n\n")

    # Insight 4 (Max House Prices)
    file.write("Insight 4: Max House Prices:\n")
    file.write("Max House Prices: " + str(df["price"].max()) + "\n\n")

    # Insight 5 (Average Prices by Preferred Area and Furnishing Status)
    file.write("Insight 5: Average Prices by Preferred Area and Furnishing Status:\n")
    file.write(average_price_by_area_and_furnishing.to_string() + "\n\n")

    # Insight 6 (Add more insights if needed)

print("Insights have been saved to 'eda-in-1.txt'.")


# In[ ]:




