import pandas as pd

def data_for_analysis(df,granularity='M'):
    '''granularity M: Monthly  W: Weekly
       Returns a dataframe (date, No. of Accidents) that represents a time series'''
    if granularity=='W':
        g='W-MON'
    else:
        g='M'

    test = pd.DataFrame(df.set_index('date').resample(g).size())
    test.columns = ['Accidents']
    return test
