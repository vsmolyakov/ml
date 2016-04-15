import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import json
from pandas.io.json import json_normalize
from hmmlearn import hmm

from datetime import datetime

if __name__ == "__main__":
        
    data_file1 = './data/intraday_steps.json'
    data_file2 = './data/intraday_heart.json'    
    with open(data_file1) as user_data:
        data1 = json.load(user_data)    
    print data1.keys()
    with open(data_file2) as user_data:
        data2 = json.load(user_data)    
    print data2.keys()    
            
    data1 = json_normalize(data1['activities-steps-intraday'])               
    data2 = json_normalize(data2['activities-heart-intraday'])     
        
    #value: 1 ("asleep"), 2 ("awake"), 3 ("really awake")
    #slp_rate = pd.DataFrame(data['minuteData'][0])    
    stp_data = pd.DataFrame(data1['dataset'][0])
    hrt_data = pd.DataFrame(data2['dataset'][0])

    #year, month, day, hour, minute, second, microsecond    
    start = datetime(2016, 4, 12, 0, 0, 0, 0)
    end = datetime(2016, 4, 12, 23, 59, 0, 0)
    rng = pd.date_range(start, end, freq='Min')

    #init time-series and assign observations
    ts = pd.Series(np.nan*np.ones(len(rng)), index=rng)
    time_idx1 = pd.to_datetime('2016-04-12 ' + stp_data['time'])
    ts[time_idx1] = stp_data['value']
        
    time_idx2 = pd.to_datetime('2016-04-12 ' + hrt_data['time'])
    hrt_ts = pd.Series(np.nan*np.ones(len(rng)), index=rng)
    hrt_ts[time_idx2] = hrt_data['value']
    ts = pd.concat([ts, hrt_ts], axis=1)
    
    #handle NaN values
    ts = ts.fillna(method='pad')

    #run HMM                            
    hmm1 = hmm.GaussianHMM(n_components=2, covariance_type='full', n_iter=100)
    hmm1.fit(ts.values)  #EM    
    hmm_states = hmm1.predict(ts.values)        

    #display learned parameters    
    print "start prob: ", hmm1.startprob_
    print "hmm_states: ", hmm_states
    print "hmm transitions: ", hmm1.transmat_
    for i in range(hmm1.n_components):
        print "mean %d: " %i
        print hmm1.means_[i]
        print "covariance %d: "%i
        print hmm1.covars_[i]
            
    #generate plots
    f, (ax1, ax2) = plt.subplots(2, sharex = True)        
    ax1.plot(ts.iloc[:,1], color='r', lw=2.0, label='heart rate')
    ax1.plot(ts.iloc[:,0], color='g', lw=2.0, label='steps')        
    ax1.set_title('HMM, Fitbit Data, 04/12/2016')
    ax1.set_ylabel('observations')
    ax1.legend(loc=0)
    ax1.set_xlim([0,1440])    
    ax1.grid(True)
            
    ax2.plot(hmm_states, color='b', lw=1.5, linestyle = '-', label='HMM state')
    ax2.set_ylabel('inferred state')
    ax2.legend(loc=0)
    ax2.grid(True)
    ax2.set_ylim([-0.1,1.1])
    ax2.set_xlim([0,1440])
    ax2.set_xlabel('time, minutes')    
    plt.show()    
    f.savefig('./fitbit_hmm.png')        
    
    