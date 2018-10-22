#Compare different algorithms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#reading data
data = pd.read_csv('brain data 4.0.csv')
#Compare algorithms results and plot it
def data_contrast(newrow_name,row_name):
    data_mean = np.mean(data[row_name])
    data_std = np.std(data[row_name])
    data_con = (data[row_name] - data_mean) / data_std
    ax1 = sns.regplot(x=newrow_name, y=data_con, data=data, x_estimator=np.mean)
    ax1.set_ylim(-3, 3)

#The age of the data was divided into different intervals
def new_category(number):
    # number is interval range
    max_number = int(90 / number) + 1
    data['new'] = 'none'
    lower_limit = 0
    for i in range(max_number):
        data.loc[data[data['age'] >= lower_limit].index, 'new'] = lower_limit
        lower_limit = lower_limit + number
    print(data)

#The column names to be compared
rowlist = ['entropy', 'variance', 'rms', 'sum_abv']
number = input('Please enter a number (space):')
new_category(int(number))
row_name = 'new'
for row in rowlist:
    data_contrast(row_name, row)
    #plt.savefig(row+'.png')
    plt.show()