
# Convolutional Auto-Encoder for Time Series Anomaly Detection

### 1. Download [REFIT Dataset](https://pureportal.strath.ac.uk/files/52873459/Processed_Data_CSV.7z)
### 2. Place _House\_{num}.csv_ files in _"./\_data/refit/"_ directory
### 3. In `preprocess.py`, initialize `house_num` and `appliance_name`
### 4. Run `preprocess.py`
### 5. Run `cae_anomaly_detection.py`

***Note: images will be save in "./_data/refit/" directory***

First, we convert the data into equal time intervals by resampling. To fill the missing data between intervals _where the resolution of sampling is lower that resampling_, we use the previous data for a limited number of intervals, and for the rest of the missing data (assuming the sensors was offline), zero is replaced. 
This limited number is calculated as a ratio of the average sampling interval in the original data.

```python

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
```
For devices that are used based on the user's need, the signals related to each usage must be separated first.
The pattern of power consumption in some appliances (for example, dishwasher and washing machine) at one usage may 
include turning the appliance on/off several times at approximately the same time intervals. 
For this reason, it may be difficult to separate the signals for different uses of the device, 
and the data for single usage may be interpreted as separate usage events.
In order to properly separate the data related to each usage event, we consider a certain period of time 
as the maximum possible time of turning off the device during one use.
Based on this value, we apply a moving window on the time series data and calculate the sum of power consumption on it. 
The usage event start when the calculated value changes from zero to a positive value.
```python
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
```

Also, we assume a maximum operation time of a device, due to the probability of a problem in the devices, which may lead to the device being continuously on 
for a very long period of time. After separating, we list the signals for each usage events, concatenate the list,
and remove noises.

the result of this process for dishwasher in house #1 is plotted here:

<img src="./_data/refit/Dishwasher/1/Power%20consumption%20of%20Dishwasher%20usage%20for%20house%20%231.png">



