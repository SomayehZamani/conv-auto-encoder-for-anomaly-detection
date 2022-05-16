
import pandas as pd
from os.path import exists
from dataset import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


if __name__ == '__main__':
    saveImage=True
    house_num=1
    appliance_name='Dishwasher'

    dataset=Dataset()
    resultDir=Dataset.getDataDir()+f'/{appliance_name}/{house_num}/'
    Path(resultDir).mkdir(parents=True, exist_ok=True)

    series=dataset.getHouseDataFrame(house_num)[appliance_name]
    # define resampling in second

    resampleSec = 10
    # compute average of sampling in sec
    mean_sample_timestamp_sec = series.reset_index()['datetime'].diff().mean().seconds

    # after resampling, we limit the fill of missing data as a ratio of time-stamp average
    fill_limit_Ratio=4 # ratio allowed to fill missing data with previous one

    # compute number of step to fill missing data according to
    # new time-stamp (resamplingSec)
    fillLimit=int(mean_sample_timestamp_sec * fill_limit_Ratio / resampleSec)

    # resample series,
    # use first value in case of existing multiple data in each new time-stamp
    # fill limited number of consecutive missing data with previous one
    # fill rest of missing data with zero
    series=series.resample(f'{resampleSec}s')\
        .first()\
        .ffill(limit=fillLimit)\
        .fillna(0)

    # define maximum duration (in seconds) of each appliance usage
    max_sec_per_event = 2 * 60 * 60

    # maximum length of data vector for each appliance usage
    max_len_per_event = int(max_sec_per_event / resampleSec)

    # define power usage threshold between On/Off states
    appliance_on_threshold=1000
    series[series<appliance_on_threshold]=0

    # define maximum duration (in seconds) the device could be off, during usage
    max_off_sec= 60 * 15

    window=int(max_off_sec / resampleSec)

    win_sum = series.rolling(window).sum()
    # convert pandas Series to DataFrame
    df = win_sum.to_frame(appliance_name)

    # compute the start of each event

    df['diff_shift'] = df[appliance_name].diff().shift(-1)

    # appliance is starting in time when it is switch between
    # off and on in rolling window.
    is_off = df[appliance_name] == 0
    become_on_next_timestamp = df['diff_shift'] > 0
    start_event_filter = is_off & become_on_next_timestamp
    df['start'] = 0
    df['start'][start_event_filter] = 1
    df['event_index'] = df['start'].cumsum()
    df['event_index'][df[appliance_name] == 0] = 0


    events_count = df['event_index'].max()
    # separating each usage data and adding to the list
    event_data_list = []
    for i in range(events_count):
        event_data = series[df['event_index'] == (i + 1)]
        # limiting duration of event if it took more than maximum limit
        if event_data.index.size > max_len_per_event:
            event_data= event_data.iloc[:max_len_per_event]
        event_data_list.append(event_data.copy())

    # extract power data for each event
    power_data_per_events = [data.reset_index()[appliance_name] for data in event_data_list]

    # extract the starting date-time of each event
    indexes=[data.reset_index()['datetime'].iloc[0] for data in event_data_list]

    # convert list of power data into one DataFrame
    power_df = pd.DataFrame(power_data_per_events).transpose()

    # use the starting time as a name of each event
    power_df.columns = indexes

    # thresholding the power-usage

    # first compute the median of total energy usage of each event as normal energy usage
    # and filter the events that use less than a fraction of normal power-consumption

    fraction_limit=0.2 # filter the events with less than 20% power-consumption of normal
    dfsum = power_df.sum()
    thereshold = dfsum.median() * fraction_limit

    filter=dfsum[dfsum > thereshold]

    # apply filter
    df_clean=power_df[filter.index.values]
    df_clean.columns=[dt.strftime("%y-%m-%d") for dt in pd.to_datetime(filter.index.values).to_pydatetime()]


    # plot power consumption
    title=f'Power consumption of {appliance_name} usage for house #{house_num}'
    ax=sns.heatmap(df_clean.fillna(0).transpose(), robust=True)
    plt.title(title)
    ax.set_ylabel('Usage date')
    ax.set_xlabel(f'time (in {resampleSec} sec)')
    if saveImage:
        plt.savefig(f'{resultDir}/{title}.png', dpi=300)
    plt.show()

    # plot cumulative energy consumption
    title=f'Cumulative energy consumption of {appliance_name} usage for house #{house_num}'
    grid_kws = {"height_ratios": (.9, .05), "hspace": .5}
    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
    ax = sns.heatmap(df_clean.fillna(0).cumsum().transpose(), robust=True, ax=ax,
                     cbar_ax=cbar_ax,
                     cbar_kws={"orientation": "horizontal"})
    ax.set_ylabel('Usage date')
    ax.set_xlabel(f'time (in {resampleSec} sec)')
    ax.set_title(title)
    if (saveImage):
        plt.savefig(f'{resultDir}/{title}.png', dpi=300)
    plt.show()

    # save data frame for using in auto-encoder
    df_clean.fillna(0).to_pickle(resultDir+'/usage_events.pkl')
