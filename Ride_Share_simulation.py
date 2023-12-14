# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 13:05:07 2023

@author: Najmaddin Akhundov


README:
  In this code we are using simulation to generate data for RIDE SHARE project.
  
  INPUTS: 
      For each run you may need to change the following:
      - The path or name of data csv file.
      - mean and std for the normal distribution used to generate departure time
      - alpha, which shows the persentage of the residents will be used in this simulatiuon
      - beta, which shows the percentage of the houses that has their own car
      - gamma, which shows the percentage of the houses with car that wants to be a flexible driver
   
  OUTPUT:
      The final results are stored in the dictionary with the name "selected_house_dict".
"""
import csv
import numpy as np
import random
import pandas as pd
random.seed(40)
np.random.seed(40)  # for fixing random seed


#------------------------------------------------------------------------------
# READ THE DATA
def read_csv_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader)  # Get the header row
        for row in csv_reader:
            # Convert values to float, skipping the first (title) column
            converted_row = {header: float(value) for header, value in zip(headers[1:], row[1:])}
            data.append(converted_row)
    return data

file_path = 'evacueeinfo/Zone_Coastal_Allinfo.csv'   # the file location should be same as the code location
csv_data = read_csv_file(file_path)

house_dict = {}   # it will store the data for each OID
house_number = 1
for row in csv_data:
    house_dict[house_number] = row
    house_number += 1

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# GENERATE RESIDENTAL DATA (we assume Poisson distribution - it is general practice)
res_per_house = 2.51    # Average number of residents per house

total_residents = 0     # it will store total resident count
for key in house_dict.keys():
    resident_count = np.random.poisson(res_per_house)   
    house_dict[key]['resident_count'] = resident_count
    total_residents += resident_count
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# GENERATE DEPARTURE TIME (we assume normal distribution)
departure_low = 1     # average time (in minutes) spent for preperation to leave the house right after the disaster
departure_up = 720       # standard deviation of departure time (in minutes)

# generate departure time for each OID
temp_departure_time = np.random.uniform(departure_up, departure_low,len(house_dict)+1)
for key in house_dict.keys():
    departure_time = round(temp_departure_time[key])
    house_dict[key]['departure_time'] = departure_time
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# RONDOM SELECTION OF RESIDENTS TO INVOLVE IN RIDE SHARING 
# Note: we can assume that selecting 10% of the residents is approximately same as slecting 10% of houses.
alpha = 0.03  # percentage of the residents that will be simulated

def generate_random_labels(num_labels, num_samples):
    labels = np.arange(1, num_labels + 1)
    random_indices = np.random.choice(labels, size=num_samples, replace=False)
    return random_indices

num_houses = len(house_dict)             # number of total houses we have
num_houses_for_sim = round(num_houses * alpha)  # number of houses we will use in simulation
random_house_selection = generate_random_labels(num_houses, num_houses_for_sim)  # it shows which houses we picked for the simulation

# generate a new dictionary and store this information
selected_house_dict = []   # it will be a subset of house_dict. It represents selected houses for the simulation.
for OID in random_house_selection:
    selected_house_dict.append(house_dict[OID])
#------------------------------------------------------------------------------
num = len(selected_house_dict)
df = pd.DataFrame(selected_house_dict)
#------------------------------------------------------------------------------
# ASSIGN CAR TO THE HOUSEHOLD
beta = 0.60   # this shows the percentage of the houses that have their own car.

num_cars = round(num_houses_for_sim * beta)  # number of houses that has a car
random_car_assignment = random.sample(list(random_house_selection), num_cars)  # it will give us OID of the houses that has a car
df['type'] = 0 #passenger

# add this information to our dictionary
for ele in random_car_assignment:
    df.loc[df['household_ID'] == ele, 'type']=1 # driver

#------------------------------------------------------------------------------
print(df.head(10))
#------------------------------------------------------------------------------
# LABEL SOME OF THE CARS AS FLEXIBLE DRIVER
gamma = 0.40   # percentage of the houses with car that are willing to be a flexible driver

num_flex_driver = round( num_cars * gamma)    # finds the number of flexible drivers
random_flexible_driver_assigment = random.sample(random_car_assignment, num_flex_driver) 

# add this information to our dictionary
for ele in random_flexible_driver_assigment:
    df.loc[df['household_ID'] == ele, 'type'] = 2 # flexible driver

#------------------------------------------------------------------------------
df.head()
#------------------------------------------------------------------------------
# SOME CALCULATIONS
total_simulated_residents = 0  # it will be used to calulate total number of residents used in the simulation
for index, row in df.iterrows():
    total_simulated_residents += row['resident_count']
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# PRINT SOME OF THE RESULTS
print('-'*50)
print('HERE ARE SOME OF THE RESULTS:\n')
print('   Total Number of Residents in the Region:', total_residents)
print('   Total Number of Residents in Considered in the Simulation:', total_simulated_residents)
print('   Total Number of Houses with Their Own Car:', num_cars)
print('   Total Number of Flexible Drivers:', num_flex_driver)
print('')
print("All Results are stored in the dicitionary named as 'selected_house_dict' \n")
print('Example:')
#for key, value in list(selected_house_dict.items())[:1]:
 #   print('For house number:',f"{key}")
  #  for k,v in list(value.items()):
   #     print(f"{k}: {v}")
        

print('-'*50)
#------------------------------------------------------------------------------
print(list(df.columns.values))
print(len(df))
column_list = ['household_ID','x_cord','y_cord','road_ID','Distance','resident_count','departure_time','type']

df[column_list].to_csv('evacueeinfo/Zone_Coastal_eva{n}_simulation_a{a}_b{b}_g{g}.csv'.format(n=num,a=alpha, b=beta, g = gamma))