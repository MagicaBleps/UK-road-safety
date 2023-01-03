
# UK Road Safety

This project was developed as final project in the Data Science Bootcamp at Le Wagon.
The aim was to realize a geographical map of road safety in the UK, using data published by the UK government here: https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data.




## Aim of the Project

The original aim of the project was to build a prediction model, using a Recurrent Neural Network (RNN), to be able to predict the future accidents rate, based on the historical data.
We wanted to divide the entire UK territory into "districts", using the [geohash encoding](https://en.wikipedia.org/wiki/Geohash), and have the model generate predictions for each single district.

This approach proved to be too ambitious for a 1 week and a half project, so we focused on the districts where accidents were more frequent. The procedure can be gradually extended to other districts, however, a separate model was trained for each district instead of having a one-model-fits-all paradigm.


## About the Dataset

We used the .csv file Road Safety Data - Accidents 1979 - 2021.
This file is a list of all recorded road accidents since 1979.
Every accident is recorded with a set of features, of which we focused primarily on:
- GPS coordinates (only for accidents occurred startng from 1999)
- timestamp
The file contains many other interesting fetures, that we didn't consider given the limited time at our disposal for the project.


## Notebooks Folder

In this folder you will find the Jupyter notebooks we used to make our analysis and create our models.

- data_analysis.ipynb: this notebook contains the preliminary analysis we did on the data. We generated some useful charts to help contextualizing the dataset and find interesting patterns.
- arima.ipynb: this notebook contains a forecasting model on the time series of monthly car accidents, using the ARIMA algorithm. It is an interesting exercise that helped us to better understand our data and to verify firsthand the effects of the 2020 pandemic on the data.
- geohash_model.ipynb: this notebook was used to generate the first Recurrent Neural Network model focused on the geohash that counted the most accidents overall (located over Trafalgar Suare in London). The same identical approach can be replicated to generate models for any other geohash.
- API_Docker_Linnan: this notebook contains all the steps performed to put the model online as an API.


## uk_road_safety Folder

This folder contains the python code we used during the project.

### python

This sub-folder contains all the custom modules we built:
- data_cleaning.py: contains all the necessary code to clean the raw data downloaded from the source website
- grouped_data.py: contains the code necessary to transform the raw data into a time series
- maps.py: contains useful code for generating interesting views of the data from the geographical point of view
- mlmodel.py: contains all the code used to prepare the data and create the RNN model

### API

This sub-folder contains the python code we used to create the API we published.
The API has two calls:
- predict: it will predict the monthly accidents rate for the first 6 months of 2022 for a given geohash
- show_map: it loads an html file containing a map of the UK with a pin point for every accident occurred in a given year, located on the accident's GPS coordinates
## Connections to Other Repositories

The [frontend repository](https://github.com/MagicaBleps/uk-road-safety-frontend) contains all the frontend code we used for the web app we published to make the model publicly available (for a limited time).

## Authors

- [@MagicaBleps](https://www.github.com/MagicaBleps)
- [@Jigisha-p](https://www.github.com/Jigisha-p)
- [@leochen888](https://www.github.com/leochen888)
- [@Bolu-Adekanbi](https://www.github.com/Bolu-Adekanmbi)


