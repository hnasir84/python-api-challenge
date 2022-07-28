## python-api-challenge

# WeatherPy

# In this section, you'll create a Python script to visualize the weather of 500+ cities of varying distance from the equator. To do so, you'll use a simple Python library, the OpenWeatherMap API, and your problem-solving skills to create a representative model of weather across cities

# Dependencies and Setup
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import time
from scipy.stats import linregress

# Import API key
from api_keys import weather_api_key

# Incorporated citipy to determine city based on latitude and longitude
from citipy import citipy

# Output File (CSV)
output_data_file = "output_data/cities_weather.csv"

# Range of latitudes and longitudes
lat_range = (-90, 90)
lng_range = (-180, 180)


# Generate Cities List

# List for holding lat_lngs and cities
lat_lngs = []
cities = []

# Create a set of random lat and lng combinations
lats = np.random.uniform(lat_range[0], lat_range[1], size=1500)
lngs = np.random.uniform(lng_range[0], lng_range[1], size=1500)
lat_lngs = zip(lats, lngs)

# Identify nearest city for each lat, lng combination
for lat_lng in lat_lngs:
    city = citipy.nearest_city(lat_lng[0], lat_lng[1]).city_name
    
    # If the city is unique, then add it to a our cities list
    if city not in cities:
        cities.append(city)

# Print the city count to confirm sufficient count
len(cities)

# Perform API Calls
Perform a weather check on each city using a series of successive API calls.
Include a print log of each city as it'sbeing processed (with the city number and city name)

# set url and unit measurement

url = "http://api.openweathermap.org/data/2.5/weather?"
city = "kodiak"
units = "imperial"

# Build query URL
queryURL = f"{url}appid={weather_api_key}&units={units}&q={city}"

# Get weather data
weatherResponse = requests.get(queryURL)
weatherJSON = weatherResponse.json()

# set empty lists to grab The required pieces of data
# identify the path to each data inside the retreived json

City =[]
Latitude = [] # weatherJSON['coord']['lat']
longtitude = [] # weatherJSON['coord']['lon']
Max_Temperature = [] # weatherJSON['main']['temp_max']
Humidity = [] # # weatherJSON['main']['humidity']
Cloudiness = [] # weatherJSON['clouds']['all']
wind_Speed = [] # weatherJSON['wind']['speed']
Country = []      # weatherJSON['sys']['country']
DateTime = [] # weatherJSON['dt']

# create counter variable to be able to group cities in groups of 50 as requested

rec_count =1
set_count =1


# loop through the cities and grab the information needed for the lists
for h, city in enumerate(cities):
    
    # group cities in group of 50s
    
    if( h % 50 == 0 and h>= 50):
       
    # rest the group every 50 counts of city
        set_count +=1
        rec_count =1
    
    # create end point url
    queryURL = f"{url}appid={weather_api_key}&units={units}&q={city}"
    
    # print the log record as data is being retrieved
    print(f" processing record {rec_count} of set {set_count} | {city}")
    rec_count +=1
    
    try:
        
        
        
        
      
           # Access and retrieve the weather data
        weatherResponse = requests.get(queryURL)
        weatherJSON = weatherResponse.json()
    
 
       # Add a one second interval between queries to stay within API query limits
        time.sleep(1)
    
    # add the info to the lists
    # add exception rule incase data was not found
        
        
        
        City.append(city)   
        Latitude.append(weatherJSON['coord']['lat'])
        longtitude.append(weatherJSON['coord']['lon'])
        Max_Temperature.append(weatherJSON['main']['temp_max'])
        Humidity.append(weatherJSON['main']['humidity'])
        Cloudiness.append(weatherJSON['clouds']['all'])
        wind_Speed.append(weatherJSON['wind']['speed'])
        Country.append(weatherJSON['sys']['country'])
        DateTime.append(weatherJSON['dt'])
        City.append(city)
      
          
    
        
        
    
    except :
        print("Missing field/result... skipping.")
        pass 
        

#  show the data retrival is complete
len(City)
#print("--------------------------------------\n \t Data Retrival is Complete \n --------------------------------------")


# Create a dictionary to hold cities weather data



cities_weather = { "City":City,
                   "Lat": Latitude,
                   "Lng":longtitude,
                   "Max Temp":Max_Temperature,
                   "Humidity":Humidity,
                   "Cloudiness":Cloudiness,
                   "Wind Speed":wind_Speed,
                   "Country":Country,
                   "Date": DateTime}

# Create data frame with the obtained list
cities_weather_DF = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in cities_weather.items()]))

Convert Raw Data to DataFrame¶
Export the city data into a .csv.
Display the DataFrame

# clean the dataframe by droping na

cities_weather_DF = cities_weather_DF.dropna()
cities_weather_DF.head()

#
cities_weather_DF.to_csv("output_data/cities_weather.csv", index=False, header=True)

# Analyze the dataframe

cities_weather_DF.head().describe()


## Inspect the data and remove the cities where the humidity > 100%.
# Skip this step if there are no cities that have humidity > 100%

# check if any city has humidity of > 100%

cities_weather_DF_humidity_check = cities_weather_DF[cities_weather_DF["Humidity"] >100]
cities_weather_DF_humidity_check

#  Get the indices of cities that have humidity over 100%.
cities_weather_DF[cities_weather_DF["Humidity"] >100].value_counts()

# show the cleaned filtered  data
cleaned_citites_weather_DF = cities_weather_DF[cities_weather_DF["Humidity"]<100]
cleaned_citites_weather_DF.head()


## Plotting the Data
# Use proper labeling of the plots using plot titles (including date of analysis) and axes labels.
# Save the plotted figures as .pngs.

# Latitude vs. Temperature Plot
Xvalues = cleaned_citites_weather_DF['Lat']
Yvalues = cleaned_citites_weather_DF['Max Temp']
plt.scatter(Xvalues,Yvalues,marker="o" , edgecolors="black")
plt.title("City Latitude vs. Max Temperature (7/26/2022)")
plt.xlabel("Latitude")
plt.ylabel("Max Temperature (F)")
plt.grid(True)
plt.savefig("output_data/Fig1_City Latitude vs. Max Temperature.png")
plt.show()


# Latitude vs. Humidity Plot
Xvalues = cleaned_citites_weather_DF['Lat']
Yvalues = cleaned_citites_weather_DF['Humidity']
plt.scatter(Xvalues,Yvalues,marker="o" , edgecolors="black")
plt.title("City Latitude vs. Humidity (7/26/2022)")
plt.xlabel("Latitude")
plt.ylabel("Humidity (%)")
plt.grid(True)
plt.savefig("output_data/Fig2_City Latitude vs. Humidity.png")
plt.show()


# Latitude vs. Cloudiness Plot
Xvalues = cleaned_citites_weather_DF['Lat']
Yvalues = cleaned_citites_weather_DF['Cloudiness']
plt.scatter(Xvalues,Yvalues,marker="o" , edgecolors="black")
plt.title(" City Latitude vs. Cloudiness (7/26/2022)")
plt.xlabel("Latitude")
plt.ylabel("Cloudiness (%)")
plt.grid(True)
plt.savefig("output_data/Fig3_City Latitude vs. Cloudiness.png")
plt.show()


# Latitude vs. Wind Speed Plot
Xvalues = cleaned_citites_weather_DF['Lat']
Yvalues = cleaned_citites_weather_DF['Wind Speed']
plt.scatter(Xvalues,Yvalues,marker="o" , edgecolors="black")
plt.title("Latitude vs. Wind Speed (7/26/2022)")
plt.xlabel("Latitude")
plt.ylabel("Wind Speed (mph)")
plt.grid(True)
plt.savefig("output_data/Fig4_Latitude vs. Wind Speed Plot.png")
plt.show()



## Linear Regression


# Northern Hemisphere - Max Temp vs. Latitude Linear Regression

# filter the northern hemisphere by lat value > 0
Northern_Hemisphere_cities = cleaned_citites_weather_DF[cleaned_citites_weather_DF['Lat']>=0]

# Add the linear regression equation and line to plot

Xvalues = Northern_Hemisphere_cities['Lat']
Yvalues = Northern_Hemisphere_cities['Max Temp']

(slope, intercept, rvalue, pvalue, stderr) = linregress(Xvalues, Yvalues)
regress_values = Xvalues * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))

plt.scatter(Xvalues,Yvalues,marker="o" , edgecolors="black")

plt.plot(Xvalues,regress_values,"r-")
plt.annotate(line_eq,(6,30),fontsize=15,color="red")



plt.xlabel("Latitude")
plt.ylabel("Max Temperautre(F)")

print(f"The r-squared is: {rvalue**2}")

plt.savefig("output_data/Fig5_Northern Hemisphere - Max Temp vs. Latitude Linear Regression.png")
plt.show()


# Southern Hemisphere - Max Temp vs. Latitude Linear Regression
# filter the northern hemisphere by lat value < 0
Southern_Hemisphere_cities = cleaned_citites_weather_DF[cleaned_citites_weather_DF['Lat']<0]

# Add the linear regression equation and line to plot

Xvalues = Southern_Hemisphere_cities['Lat']
Yvalues = Southern_Hemisphere_cities['Max Temp']

(slope, intercept, rvalue, pvalue, stderr) = linregress(Xvalues, Yvalues)
regress_values = Xvalues * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
plt.scatter(Xvalues,Yvalues,marker="o" , edgecolors="black")

plt.plot(Xvalues,regress_values,"r-")
plt.annotate(line_eq,(4,1),fontsize=10,color="red")



plt.xlabel("Latitude")
plt.ylabel("Max Temperature (F)")

print(f"The r-squared is: {rvalue**2}")
plt.savefig("output_data/Fig6_Southern Hemisphere - Max Temp vs. Latitude Linear Regression.png")
plt.show()


# Northern Hemisphere - Humidity (%) vs. Latitude Linear Regression

# filter the northern hemisphere by lat value > 0
Northern_Hemisphere_cities = cleaned_citites_weather_DF[cleaned_citites_weather_DF['Lat']>=0]

# Add the linear regression equation and line to plot

Xvalues = Northern_Hemisphere_cities['Lat']
Yvalues = Northern_Hemisphere_cities['Humidity']

(slope, intercept, rvalue, pvalue, stderr) = linregress(Xvalues, Yvalues)
regress_values = Xvalues * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
plt.scatter(Xvalues,Yvalues,marker="o" , edgecolors="black")

plt.plot(Xvalues,regress_values,"r-")
plt.annotate(line_eq,(4,1),fontsize=15,color="red")



plt.xlabel("Latitude")
plt.ylabel("Humidity (%)")

print(f"The r-squared is: {rvalue**2}")

plt.savefig("output_data/Fig7_Northern Hemisphere - Humidity (%) vs. Latitude Linear Regression.png")
plt.show()


# Southern Hemisphere - Humidity (%) vs. Latitude Linear Regression

# filter the northern hemisphere by lat value < 0
Southern_Hemisphere_cities = cleaned_citites_weather_DF[cleaned_citites_weather_DF['Lat']<0]

# Add the linear regression equation and line to plot

Xvalues = Southern_Hemisphere_cities['Lat']
Yvalues = Southern_Hemisphere_cities['Humidity']

(slope, intercept, rvalue, pvalue, stderr) = linregress(Xvalues, Yvalues)
regress_values = Xvalues * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
plt.scatter(Xvalues,Yvalues,marker="o" , edgecolors="black")

plt.plot(Xvalues,regress_values,"r-")
plt.annotate(line_eq,(4,1),fontsize=10,color="red")


plt.title("Latitude vs. Humidity (7/26/2022)")
plt.xlabel("Latitude")
plt.ylabel("Humidity (%)")

print(f"The r-squared is: {rvalue**2}")
plt.savefig("output_data/Fig8_Southern Hemisphere - Humidity (%) vs. Latitude Linear Regression.png")
plt.show()


# Northern Hemisphere - Cloudiness (%) vs. Latitude Linear Regression

# filter the northern hemisphere by lat value > 0
Northern_Hemisphere_cities = cleaned_citites_weather_DF[cleaned_citites_weather_DF['Lat']>=0]

# Add the linear regression equation and line to plot

Xvalues = Northern_Hemisphere_cities['Lat']
Yvalues = Northern_Hemisphere_cities['Cloudiness']

(slope, intercept, rvalue, pvalue, stderr) = linregress(Xvalues, Yvalues)
regress_values = Xvalues * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
plt.scatter(Xvalues,Yvalues,marker="o" , edgecolors="black")

plt.plot(Xvalues,regress_values,"r-")
plt.annotate(line_eq,(4,1),fontsize=10,color="red")



plt.xlabel("Latitude")
plt.ylabel("Cloudiness")

print(f"The r-squared is: {rvalue**2}")
plt.savefig("output_data/Fig9_Northern Hemisphere - Cloudiness (%) vs. Latitude Linear Regression.png")
plt.show()


# Southern Hemisphere - Cloudiness (%) vs. Latitude Linear Regression

# filter the northern hemisphere by lat value < 0
Southern_Hemisphere_cities = cleaned_citites_weather_DF[cleaned_citites_weather_DF['Lat']<0]

# Add the linear regression equation and line to plot

Xvalues = Southern_Hemisphere_cities['Lat']
Yvalues = Southern_Hemisphere_cities['Cloudiness']

(slope, intercept, rvalue, pvalue, stderr) = linregress(Xvalues, Yvalues)
regress_values = Xvalues * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
plt.scatter(Xvalues,Yvalues,marker="o" , edgecolors="black")

plt.plot(Xvalues,regress_values,"r-")
plt.annotate(line_eq,(4,1),fontsize=10,color="red")


plt.title("Latitude vs. Cloudiness (7/26/2022)")
plt.xlabel("Latitude")
plt.ylabel("Cloudiness")

print(f"The r-squared is: {rvalue**2}")
plt.savefig("output_data/Fig10_Southern Hemisphere - Cloudiness (%) vs. Latitude Linear Regression.png")
plt.show()


# Northern Hemisphere - Wind Speed (mph) vs. Latitude Linear Regression

# filter the northern hemisphere by lat value > 0
Northern_Hemisphere_cities = cleaned_citites_weather_DF[cleaned_citites_weather_DF['Lat']>=0]

# Add the linear regression equation and line to plot

Xvalues = Northern_Hemisphere_cities['Lat']
Yvalues = Northern_Hemisphere_cities['Wind Speed']

(slope, intercept, rvalue, pvalue, stderr) = linregress(Xvalues, Yvalues)
regress_values = Xvalues * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
plt.scatter(Xvalues,Yvalues,marker="o" , edgecolors="black")

plt.plot(Xvalues,regress_values,"r-")
plt.annotate(line_eq,(4,1),fontsize=15,color="red")


plt.title("Latitude vs. Wind Speed (7/26/2022)")
plt.xlabel("Latitude")
plt.ylabel("Wind Speed")

print(f"The r-squared is: {rvalue**2}")
plt.savefig("output_data/Fig11_Northern Hemisphere - Wind Speed (mph) vs. Latitude Linear Regression.png")
plt.show()

# Southern Hemisphere - Wind Speed (mph) vs. Latitude Linear Regression

# filter the northern hemisphere by lat value < 0
Southern_Hemisphere_cities = cleaned_citites_weather_DF[cleaned_citites_weather_DF['Lat']<0]

# Add the linear regression equation and line to plot

Xvalues = Southern_Hemisphere_cities['Lat']
Yvalues = Southern_Hemisphere_cities['Wind Speed']

(slope, intercept, rvalue, pvalue, stderr) = linregress(Xvalues, Yvalues)
regress_values = Xvalues * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
plt.scatter(Xvalues,Yvalues,marker="o" , edgecolors="black")

plt.plot(Xvalues,regress_values,"r-")
plt.annotate(line_eq,(4,1),fontsize=10,color="red")



plt.xlabel("Latitude")
plt.ylabel("Wind Speed")

print(f"The r-squared is: {rvalue**2}")
plt.savefig("output_data/Fig12_Southern Hemisphere - Wind Speed (mph) vs. Latitude Linear Regression.png")
plt.show()

# The end of Weatherpy code




## VacationPy

# Supress warnings
import warnings
warnings.filterwarnings('ignore')

# Dependencies and Setup
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import gmaps
import json
from pprint import pprint
import os

# Import API key
from api_keys import g_key

## Store Part I results into DataFrame¶
# Load the csv exported in Part I to a DataFrame

# read the csv file from weatherpy exercice and convert it to data frame

cities = pd.read_csv("../WeatherPy/output_data/cities_weather.csv")
cities.tail()

##  Humidity Heatmap
# Configure gmaps.
# Use the Lat and Lng as locations and Humidity as the weight.
# Add Heatmap layer to map.



# configure gmaps
gmaps.configure(api_key=g_key)

# create humudity map
locations = cities[['Lat', 'Lng']]
weight = cities["Humidity"]

# Plot Heatmap
fig = gmaps.figure()

# Create heat layer
heat_layer = gmaps.heatmap_layer(locations, weights= weight, 
                                 dissipating=False, max_intensity=10,
                                 point_radius=1)


# Add layer
fig.add_layer(heat_layer)

# Display figure
fig


## Create new DataFrame fitting weather criteria
# Narrow down the cities to fit weather conditions.
# Drop any rows will null values.

# Narrow down the DataFrame to find your ideal weather condition. For example:
# A max temperature lower than 80 degrees but higher than 70.
# Wind speed less than 10 mph.
# Zero cloudiness.

# Drop any rows that don't satisfy all three conditions. You want to be sure the weather is ideal.


ideal_vacaton_cities = cities.loc[(cities['Max Temp'] > 70) & (cities['Max Temp'] < 80) & (cities["Wind Speed"] < 10) &(cities["Cloudiness"]==0)].dropna()
ideal_vacaton_cities



## Hotel Map
# Store into variable 
# Add a "Hotel Name" column to the DataFrame.
# Set parameters to search for hotels with 5000 meters.
# Hit the Google Places API for each city's coordinates.
# Store the first Hotel result into the DataFrame.
# Plot markers on top of the heatmap.

hotels_nearby = ideal_vacaton_cities[["City", "Country", "Lat", "Lng"]]
hotels_nearby["Hotel Name"] = ""
hotels_nearby


# set the parameters



parameter = { 
            "radius": 50000,
            "types": "hotels",
            "key": g_key}
     
              
          
      

# use ierrows to loop through dataframe

for index, row in hotels_nearby.iterrows():
    
    # get the latitude and longtitude from the current row
    lat = row["Lat"]
    lng = row["Lng"]
    
    # append the obtained latitude and longtitude to params dict
    parameter["location"] = f"{lat},{lng}"
  
    
    # utilize google places api to find the hotels
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    
    
    hotel_name= requests.get(url,parameter).json()
 
   # pprint(hotel_address) #['results'][0]['name']
    
    # Apply exception rule incase and result wasn't found
    try:
        # generate hotel names to the dataframe
        hotels_nearby.loc[index,"Hotel Name"] = hotel_name['results'][0]['name']
    except:
        print(" No hotel was found in {row['City']}. skip")
        
hotels_nearby



# Store the DataFrame Row
# NOTE: be sure to update with your DataFrame name
hotel_info = [info_box_template.format(**row) for index, row in hotels_nearby.iterrows()]
locations = hotels_nearby[["Lat", "Lng"]]

# Add marker layer ontop of heat map
marker = gmaps.marker_layer(locations, info_box_content =hotel_info )
fig.add_layer(marker)

# Display figure
fig


