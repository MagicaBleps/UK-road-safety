#These functions are almost complete. I just need to touch up on some of the functionality and features. I plan to combine this along, with a few more functions to create a fully customizable data analysis interactive board.
def select_year(start_year,path="../Clean_data_1999-2021",stop_year=0):
    """
    start_year: The first scope of the data you would like to access
    path: The path to the dataset
    stop_year(optional): Combines with start_year to provide data between start and stop year inclusive
    """
    import pandas as pd
    if stop_year == 0:
        data = pd.read_csv(path)
        data = data[data['accident_year'] == start_year]
        return data
    data = pd.read_csv(path)
    year_step = [year for year in range(start_year, stop_year+1)]
    data = data[data['accident_year'].isin(year_step) ]
    return data

def basic_info(data):
    """
    Returns basic information of the input data such as: day of the week value_counts...
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, axes = plt.subplots(2,2, figsize=(10,8), sharey=True)
    sns.countplot(data['day_of_week'], ax= axes[0][0])
    #sns.countplot(data['day_of_week'], ax= axes[0][0])
    #sns.countplot(data['day_of_week'], ax= axes[0][0])
    fig.suptitle("Basic Information about accident occurrence")
    fig.tight_layout();
    fig.show();

def map_plot(small_data,column='day_of_week'):
    """
    Returns a geographical map view of accident, locations and the date of the week they took place
    """
    import plotly.express as px
    import pandas as pd
    import numpy as np

    df = small_data
    num = len(df[column].unique())
    colors = ['red', 'green', 'orange', 'yellow','pink', 'blue','black']
    values = list(np.linspace(0,1,num))
    color_scale = []
    for i in range(num):
        color_scale.append((values[i],colors[i]))

    fig = px.scatter_mapbox(df,
                            lat="latitude",
                            lon="longitude",
                            #hover_name="Address",
                            #hover_data=["Address", "Listed"],
                            color=column,
                            color_continuous_scale=color_scale,
                            #size="accident_severity",
                            zoom=8,
                            height=800,
                            width=800)

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()
