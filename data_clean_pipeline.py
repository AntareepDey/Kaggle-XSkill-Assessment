import pandas as pd

# Load the data
df = pd.read_csv("train.csv")

def data_cleaning(data):
    # Drop unnecessary columns
    data.drop('clean_title', axis=1, inplace=True)
    data.drop('model', axis=1, inplace=True)
    data.drop('id', axis=1, inplace=True)
    
    # Filter data based on price
    data = data[data['price'] < 650000]
    
    def classify_transmission(transmission):
        transmission = transmission.lower()
        if 'm/t' in transmission or 'manual' in transmission:
            return False
        else:
            return True
    
    # Apply classification to transmission column
    data['transmission_auto'] = data['transmission'].apply(classify_transmission)
    data.drop('transmission', axis=1, inplace=True)
    
    # Clean accident column
    data['accident'] = data['accident'].replace('At least 1 accident or damage reported', True)
    data.loc[data['accident'] != True, 'accident'] = False
    
    def fuelclass(fuel):
        fuel = fuel.lower()
        if fuel in ['gasoline', 'diesel']:
            return 'gas'
        elif fuel in ['hybrid', 'plug-in hybrid']:
            return 'hybrid'
        else:
            return 'other'
    
    # Apply fuel classification
    data['fuel_type'] = data['fuel_type'].apply(fuelclass)

    def paint(colour):
        colour = colour.lower()
        if colour in ['black', 'gray', 'slate', 'grey']:
            return 'black'
        elif colour in ['white', 'silver', 'beige']:
            return 'white'
        else:
            return 'other'
    
    # Apply paint classification to interior and exterior color columns
    data['int_col'] = data['int_col'].apply(paint)
    data['ext_col'] = data['ext_col'].apply(paint)
    
    def transform_engine(car_data):
        # Split engine column into separate columns
        split_engine = car_data['engine'].str.split(' ', expand=True)
        
        # Extract horsepower, litres, and cylinders information
        split_engine.loc[split_engine[0].str.contains('HP'), 'horsepower'] = split_engine.loc[split_engine[0].str.contains('HP'), 0].str.replace('HP', "").astype(float)
        split_engine.loc[(split_engine[0].str.contains('HP')) & (split_engine[1].str.contains('L')), 'litres'] = split_engine.loc[(split_engine[0].str.contains('HP')) & (split_engine[1].str.contains('L')), 1].str.replace('L', "").astype(float)
        split_engine.loc[split_engine[0].str.contains('HP') & (split_engine[3] == 'Cylinder'), 'cylinders'] = split_engine.loc[split_engine[0].str.contains('HP') & (split_engine[3] == 'Cylinder'), 2].str.replace('V', "").astype(int)
    
        
        # Merge extracted columns with original data
        car_data = pd.concat([car_data, split_engine[['horsepower', 'litres', 'cylinders']]], axis=1)

        # Drop unnecessary columns
        car_data.drop(['engine', 'hp_mean', 'hp_mean_litres', 'hp_mean_brand', 'litre_mean', 'litre_mean_hp', 'litre_mean_brand', 'cy_mean', 'cy_mean_litres', 'cy_mean_brand'], axis=1, inplace=True)

        return car_data
    
    # Apply engine transformation
    data = transform_engine(data)
    
    def categorise_brand(brand):
        luxury = ['Bugatti', 'Ferrari', 'Lamborghini', 'Rolls-Royce', 'Aston', 'McLaren', 'Bentley', 'Lucid', 'Rivian', 'Porsche', 'Maybach', 'Maserati', 'Genesis']
        midrange = ['Tesla', 'Land', 'Mercedes-Benz', 'Alfa', 'RAM', 'Chevrolet', 'GMC', 'BMW', 'Lotus', 'Ford', 'Audi', 'Cadillac', 'Jaguar']
        
        if brand in luxury:
            return 'Luxury'
        elif brand in midrange:
            return 'Midrange'
        else:
            return 'Economy'
    
    # Apply brand categorization
    data['brand'] = data['brand'].apply(categorise_brand)
    
    # Perform one-hot encoding on categorical columns
    data = pd.get_dummies(data, columns=['brand', 'int_col', 'ext_col', 'fuel_type'])

    return data

# pass the training data through the data cleaning pipeline
df = data_cleaning(df)

# Save the cleaned data to a new file
df.to_csv('cleaned_train_data.csv', index=False)
