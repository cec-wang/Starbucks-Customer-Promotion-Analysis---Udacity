'''Functions to help clean and analyze data from Starbucks dataset'''
import progressbar
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import json
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
#Visualise Clustering with TSNE
from sklearn.manifold import TSNE
#% matplotlib inline
import seaborn as sns

# read in the json files
portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('data/profile.json', orient='records', lines=True)
transcript = pd.read_json('data/transcript.json', orient='records', lines=True)


def test_print():
    print('Functions imported successfully.')


def channel_separation(data = portfolio):
    '''
    Cleans portfolio data and separate categorial columns into separate columms.
    INPUT:
    data: DataFrame of the original portfolio data
    OUTPUT:
    data: new DataFrame with channel separated and original channel column removed'''
    # get a set of all possible channels
    channels = []
    for channel in data['channels']:
        channels.extend(channel)
    channels = set(channels)

    #split the channels into separate columns
    for i, items in enumerate(data.channels):
        for channel in channels:
            if channel in items:
                data.loc[i, channel] = 1
            else:
                data.loc[i, channel] = 0
    # drop the original channels column
    data = data.drop(columns = 'channels')

    return data


def portfolio_data_cleaning(portfolio = portfolio):
    '''
    cleans the portfolio data
    INPUT:
    portfolio: DataFrame of the original portfolio data
    OUTPUT:
    portfolio: DataFrame cleaned and each categorial columns separated.
    '''
    portfolio = channel_separation(data = portfolio)
    # set the id column as index
    portfolio.set_index('id', drop = True, inplace = True)
    #separate the offer types into separate columns
    portfolio['val'] =1
    portfolio_offer = portfolio.pivot_table(values = 'val', index = 'id', columns = 'offer_type', fill_value = 0)

    portfolio = portfolio.join(portfolio_offer, on = portfolio.index)

    portfolio = portfolio.drop(columns = 'offer_type')

    # add a new row with id 'wo_offer'
    wo_offer_row = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    row_df = pd.DataFrame(wo_offer_row).T
    row_df.index = ['wo_offer']
    row_df.columns = portfolio.columns
    portfolio = pd.concat([portfolio, row_df])

    portfolio['no_offer'] = 0
    portfolio.loc['wo_offer', 'no_offer'] = 1

    return portfolio


def profile_clean(profile = profile):
    '''
    cleans the profile data
    INPUT:
    profile: DataFrame of the original profile data
    OUTPUT:
    profile: DataFrame cleaned and each categorial columns separated.
    '''
    # change the age 118 to the median age of the rest of the group
    print('number of users with no age entry:', profile[profile.age==118].shape[0])
    if 118 in profile.age:
        median_age = profile[profile.age!=118].age.median()
        noage_index = profile[profile.age==118].index
        profile.loc[noage_index, 'age'] = median_age

    print('number of users without salary entry:', profile.income.isna().sum(),
          '\npercentage of users without salary entry:', profile.income.isna().sum()/170, '%')
    #fill the non-filled salary with average salary
    profile.income.fillna(profile.income.mean(), inplace = True)

    # One hot encoding to separate gender into 4 different categories
    profile = pd.get_dummies(profile, dummy_na = True, columns = ['gender'], prefix = 'gender')

    #Update the became_member_on column from an integer of yearmonthdate to an integer
    #of days from today.
    today= pd.to_datetime('today')
    profile['became_member_on'] = pd.to_datetime(profile['became_member_on'], format = '%Y%m%d')
    profile['membership_days'] = today-profile['became_member_on']
    profile['membership_days'] = profile['membership_days'].transform(lambda x: x.days)
    profile = profile.drop(columns = 'became_member_on')

    #set column 'id' as index
    profile = profile.set_index('id')

    return profile


def generate_transform_transcript(transcript):
    '''
    generates the transformed transcript
    INPUT: transcript (DataFrame)
    OUTPUT: transcript_transformed (DataFrame)
    '''
    # transform the transcript to contain all possible combinations of person and offers

    people = transcript.person.unique()
    offers = list(portfolio.index)
    people_lst = []
    for person in people:
        people_lst = people_lst + [person] * 11
    offers_lst = offers * len(people)
    print (len(people_lst), len(offers_lst))
    #transcript_trans = pd.DataFrame({'person': people_lst, 'offer': offers_lst})

    return transcript_trans


def transcript_clean(transcript = transcript, portfolio = portfolio):
    '''
    cleans the transcript data
    INPUT:
    transcript: DataFrame of the original transcript data
    OUTPUT:
    transcript_transformed: DataFrame cleaned and each categorial columns separated.
    '''
     # take the brackets off the value column
    transcript.value = transcript.value.transform(lambda x: list(x.values())[0])

    print('Transforming transcript')
    people = transcript.person.unique()
    offers = list(portfolio.index)
    people_lst = []
    for person in people:
        people_lst = people_lst + [person] * 11
    offers_lst = offers * len(people)
    transcript_transformed = pd.DataFrame({'person': people_lst, 'offer': offers_lst})

    print('Assigning offer received, offer viewed and offer completed values...')

    #generate dataframes only contains certain events, grouped and counted
    offer_received_count = event_grouped_count(transcript = transcript, \
    event = 'offer received', groupby = ['person', 'value'])

    offer_viewed_count = event_grouped_count(transcript = transcript, \
    event = 'offer viewed', groupby = ['person', 'value'])

    transaction_count = event_grouped_count(transcript = transcript, \
    event = 'transaction', groupby = ['person'])

    offer_completed_count = event_grouped_count(transcript = transcript, \
    event = 'offer completed', groupby = ['person', 'value'])
    complete_people_grouped = event_grouped_count(transcript = transcript, \
    event = 'offer completed', groupby = ['person'])

    # Merge the new data to the transcript_transformed
    transcript_transformed = transcript_transformed.merge(offer_received_count[['event']], how = 'left',\
                                                      left_on = ['person', 'offer'], right_on = ['person', 'value'],\
                                                      validate = 'one_to_one')
    transcript_transformed = transcript_transformed.rename(columns = {'event':'offer_received'})

    transcript_transformed = transcript_transformed.merge(offer_viewed_count[['event']], how = 'left',\
                                                          left_on = ['person', 'offer'], right_on = ['person', 'value'],\
                                                          validate = 'one_to_one')
    transcript_transformed = transcript_transformed.rename(columns = {'event':'offer_viewed'})

    transcript_transformed = transcript_transformed.merge(offer_completed_count[['event']], how = 'left',\
                                                          left_on = ['person', 'offer'], right_on = ['person', 'value'],\
                                                          validate = 'one_to_one')
    transcript_transformed = transcript_transformed.rename(columns = {'event':'transaction'})

    print('Assigning transaction count...')
    # to calculate how may transactions are not from offers for each user
    transaction_count = transaction_count.rename({'event': 'total_trans'})
    transaction_count = transaction_count.merge(complete_people_grouped, how = 'outer', \
    left_index = True, right_index = True, validate = 'one_to_one')
    transaction_count['event_y'].fillna(0, inplace = True)
    transaction_count['diff'] = transaction_count.event_x - transaction_count.event_y
    transaction_count['offer'] = 'wo_offer'
    transcript_transformed.transaction.fillna(0, inplace = True)

    #merge the transaction number to the transcript transformed
    transcript_transformed = transcript_transformed.merge(transaction_count[['diff', 'offer']], how = 'left',\
                                                      left_on = ['person', 'offer'], right_on = ['person', 'offer'],\
                                                      validate = 'one_to_one')
    transcript_transformed['diff'].fillna(0, inplace = True)
    transcript_transformed['sum'] = transcript_transformed['diff'] + transcript_transformed['transaction']
    transcript_transformed = transcript_transformed.drop(columns = ['transaction', 'diff']).rename(columns = {'sum': 'transaction'})

    transcript_transformed['offer_received'].fillna(0, inplace = True)
    transcript_transformed['offer_viewed'].fillna(0, inplace = True)

    print('Working on transaction value now...')
    # generate new DataFrame to extract transaction amount
    offer_completed_df = transcript[transcript['event'] == 'offer completed']
    transaction_df = transcript[transcript['event'] == 'transaction']
    offer_completed_df = offer_completed_df.reset_index().drop(columns = 'index')
    transaction_df = transaction_df.sort_values(['person', 'time']).reset_index().drop(columns = 'index')

    print('Extracting transaction value')
    # Create the progressbar
    bar_num = offer_completed_df.shape[0]
    cnter = 0
    bar = progressbar.ProgressBar(maxval=bar_num+1, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    duration_dict = generate_duration_dict(portfolio)
    #loop through the offer_completed DataFrame
    for i in range(offer_completed_df.shape[0]):
        # Update the progress bar
        cnter += 1
        bar.update(cnter)
        #for each entry in offer_completed_df, find a set of entries from transaction from the transaction df
        new_df = transaction_df[transaction_df['person'] == offer_completed_df.loc[i, 'person']]
        #Find the transactions between time period of the offer
        offer = offer_completed_df.loc[i, 'value']
        offer_time = offer_completed_df.loc[i, 'time']
        new_df_timed = new_df[(new_df['time'] <= offer_time) & \
                       (new_df['time'] >  (offer_time - 24*duration_dict[offer]))]

        #Extract the transaction amount from that entry and assign it as the amount for that offer
        new_df_timed['value'] = pd.to_numeric(new_df_timed['value'])
        offer_completed_df.loc[i, 'transaction'] = new_df_timed.value.sum()

    bar.finish()
    # Calculate and assign the total amount of transaction for each offer per person
    offer_completed_df_count = offer_completed_df.groupby(['person', 'value']).sum()
    transcript_transformed = transcript_transformed.merge(offer_completed_df_count[['transaction']], how = 'left',\
                                                      left_on = ['person', 'offer'], right_on = ['person', 'value'],\
                                                      validate = 'one_to_one')

    transaction_df['value'] = pd.to_numeric(transaction_df['value'])

    offer_completed_person_sum = offer_completed_df.groupby('person').sum()
    transaction_sum = transaction_df.groupby('person').sum()

    transaction_sum = transaction_sum.merge(offer_completed_person_sum['transaction'], how = 'outer',\
                                       left_on = ['person'], right_on = ['person'],\
                                                      validate = 'one_to_one')
    # the amount of transaction made without offer was calculated by substracting the total transaction amount per person
    # with the amount from the other transactions from offers
    transaction_sum['diff'] = transaction_sum['value'] - transaction_sum['transaction']
    transaction_sum['offer'] = 'wo_offer'
    transcript_transformed = transcript_transformed.merge(transaction_sum[['diff', 'offer']], how = 'left',\
                                                      left_on = ['person', 'offer'], right_on = ['person', 'offer'],\
                                                      validate = 'one_to_one')

    print('performing final dataframe cleaning...')
    # final data cleaning
    transcript_transformed['transaction_y'].fillna(0, inplace = True)
    transcript_transformed['diff'].fillna(0, inplace = True)
    transcript_transformed['transaction_amt'] = transcript_transformed['transaction_y'] + transcript_transformed['diff']
    transcript_transformed = transcript_transformed.rename(columns = {'transaction_x':'transaction'})
    transcript_transformed.drop(columns = ['transaction_y', 'diff'], inplace = True)

    return transcript_transformed

def event_grouped_count(event, groupby, transcript = transcript):
    '''
    filter the transcript by event, then group by groupby
    INPUT:
    event: (string) the interested event
    groupby: (list of string)
    transcript: (DataFrame)
    OUTPUT:
    event_grouped: (DataFrame)filtered and grouped transcript)
    '''
    event_df = transcript[transcript['event'] == event]
    event_grouped = event_df.groupby(groupby).count()

    return event_grouped

def generate_duration_dict(portfolio):
    '''
    generates an diction with offer id as keys and offer duration as values
    '''
    duration_dict = {}
    for offer in portfolio.index:
        duration_dict[offer] = portfolio.loc[offer, 'duration']
    return duration_dict

def plot_demo_attribute(demographic, labels, lower_limit, higher_limit, step, trans_amt_master):
    '''
    plot a grouped bar graph showing the distribution of transaction amount from
    each offer attribute among certain demographic groups
    INPUT:
    demographic: (string) demographic of interest
    labels: (list) list of offer attributes
    lower_limit, higher_limit, step: (int) range of demographic character
    trans_amt_master: master matrix to extract data from
    OUTPUT:
    a grouped bar plot
    '''
    num_mem = np.arange(lower_limit, higher_limit, step)

    ind = np.arange(len(labels))
    fig, ax = plt.subplots(figsize = (18, 8))

    for i in range(len(num_mem)-1):
        data = trans_amt_master[(trans_amt_master[demographic] < num_mem[i+1]) &\
                                          (trans_amt_master[demographic] > num_mem[i])][labels].mean()
        ax.bar(x = (ind + 0.1*i) , height = data, width = 0.1, label = num_mem[i])

    ax.set_xticklabels(labels2, fontsize = 16)
    ax.set_xticks(ind)
    ax.set_ylabel('Relative total Transaction Amount', fontsize = 16);
    ax.set_title('Average Transaction Amount by {}'.format(demographic), fontsize = 16)
    ax.legend()
    #autolabel(rects1, ax = ax)

    plt.show()

def density_based_clustering(data, eps = 0.4, min_samples = 10):
    '''
    perform density based clustering on a dataset.
    INPUT:
    data: DataFrame of the data to be clustered
    eps: epsilon (float) used for clustering
    min_samples: (integer) minimal number of samples per cluster
    OUTPUT:
    labels: numpy array of labels
    clusterNum: (int) number of cluster
    realClusterNum: (int) number of real clusters
    '''
    data_fitted = StandardScaler().fit_transform(data)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data_fitted)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
    clusterNum = len(set(labels))
    return labels, clusterNum, realClusterNum


def plot_dist_distribution(data, n_neighbors=10):
    '''
    plot a distribution of distance between data points with its n nearest n_neighbors
    INPUT:
    data: dataset to be looked at
    n_neighbors: (int) number of neighbors to be calculated
    '''
    data_fitted = StandardScaler().fit_transform(data)
    neigh = NearestNeighbors(n_neighbors = n_neighbors)
    nbrs = neigh.fit(data_fitted)
    distances, indices = nbrs.kneighbors(data_fitted)

    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)

def cluster_by_epi_size(min_epi, max_epi, epi_step, sample_sizes, data):
    '''
    optimize epsilon and minumum sample size at a pre-defined range
    INPUT:
    min_epi, max_epi, epi_step: (number) range of epsilon
    sample_sizes: (list or array) minumum samples sizes to try
    data: data to perform density based clustering on

    OUTPUT:
    plot two heat maps on number of clusters generated and number of data not clustered
    '''
    # get the most reasonable epsilon and sample sizes
    num_epis = np.arange(min_epi, max_epi, epi_step)

    num_clusters = []
    non_clust_num = []

    # Create the progressbar
    bar_num = len(num_epis) * len(sample_sizes)
    cnter = 0
    bar = progressbar.ProgressBar(maxval=bar_num+1, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    #loop though each epsilon and minimum sample size to get the cluster number and number of non-clustered points matrix
    for epi in num_epis:
        cluster_row = []
        non_clust = []
        for size in sample_sizes:
            labels, clusterNum, realClusterNum = density_based_clustering(data, eps = epi, min_samples = size)
            cluster_row = cluster_row + [realClusterNum]
            non_clust = non_clust + [np.count_nonzero(labels == -1)]
            cnter += 1
            bar.update(cnter)
        num_clusters.append(cluster_row)
        non_clust_num.append(non_clust)
    bar.finish()

    #plot the heatmap
    fig, ax = plt.subplots(1, 2, figsize = (10, 10))
    im = ax[0].imshow(num_clusters)

    # We want to show all ticks...
    ax[0].set_xticks(np.arange(len(sample_sizes)))
    ax[0].set_yticks(np.arange(len(num_epis)))
    # ... and label them with the respective list entries
    ax[0].set_xticklabels(sample_sizes)
    ax[0].set_xlabel('minmun sample numbers', fontsize = 16)
    ax[0].set_ylabel('epislon number', fontsize = 16)
    # Beat them into submission and set them back again
    ax[0].set_yticklabels([round(label, 2) for label in num_epis])
    ax[0].set_title('number of Clustered Generated')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(num_epis)):
        for j in range(len(sample_sizes)):
            text = ax[0].text(j, i, num_clusters[i][j],
                           ha="center", va="center", color="w", fontsize = 16)

    # plot a heatmap of non-clustered sample numbers. We want the number of samples not clustered as little as possible
    im1 = ax[1].imshow(non_clust_num)
    ax[1].set_title("number of people not clustered per parameter")

    ax[1].set_xticks(np.arange(len(sample_sizes)))
    ax[1].set_yticks(np.arange(len(num_epis)))
    # ... and label them with the respective list entries
    ax[1].set_xticklabels(sample_sizes)
    ax[1].set_xlabel('minmun sample numbers', fontsize = 16)
    ax[1].set_ylabel('epislon number', fontsize = 16)
    # Beat them into submission and set them back again
    ax[1].set_yticklabels([round(label, 2) for label in num_epis])
    # Rotate the tick labels and set their alignment.
    plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(num_epis)):
        for j in range(len(sample_sizes)):
            text = ax[1].text(j, i, non_clust_num[i][j],
                           ha="center", va="center", color="k", fontsize = 12)
    plt.show()

def plot_demo_attribute(demographic, labels, lower_limit, higher_limit, step, trans_amt_master):
    num_mem = np.arange(lower_limit, higher_limit, step)

    ind = np.arange(len(labels))
    fig, ax = plt.subplots(figsize = (18, 8))

    for i in range(len(num_mem)-1):
        data = trans_amt_master[(trans_amt_master[demographic] < num_mem[i+1]) &\
                                          (trans_amt_master[demographic] > num_mem[i])][labels].mean()
        ax.bar(x = (ind + 0.1*i) , height = data, width = 0.1, label = num_mem[i])

    ax.set_xticklabels(labels, fontsize = 16)
    ax.set_xticks(ind)
    ax.set_ylabel('Relative total Transaction Amount', fontsize = 16);
    ax.set_title('Average Transaction Amount by {}'.format(demographic), fontsize = 16)
    ax.legend()
    #autolabel(rects1, ax = ax)

    plt.show()

def plot_cluster(data, labels):
    print('generating data...')
    data_fitted = StandardScaler().fit_transform(data)
    tsne = TSNE(random_state=42).fit_transform(data_fitted)
    tsne_df = pd.DataFrame(tsne, columns=['xs', 'ys'])
    tsne_df['cluster'] = ['cluster ' + str(i) for i in labels]

    #plot tsne
    print('plotting data...')
    fig, ax = plt.subplots(figsize=(10,8))
    sns.scatterplot('xs','ys', data=tsne_df, ax=ax, hue='cluster' )
    ax.set_title("Cluster Visualisation with TSNE")
    ax.set_xlim([-100, 100])
    ax.set_ylim([-100, 100])

    plt.show()

def plot_cluster_data(data, cluster_labels, labels, clusterNum, all_real = True):
    data_clustered = data.copy()
    data_clustered['clusters'] = cluster_labels
    data_grouped = data_clustered.groupby('clusters').mean()

    fig, ax = plt.subplots((len(labels)//2 +1), 2, figsize = (10, 20))
    for i in range(len(labels)):
        data = data_grouped[labels[i]]
        if all_real:
            ax[i//2, i%2].bar(x = np.arange(0, clusterNum), height = data, width = 0.5)
        else:
            ax[i//2, i%2].bar(x = np.arange(-1, clusterNum-1), height = data, width = 0.5)

        ax[i//2, i%2].set_title(labels[i], fontsize = 16)
plt.show()


print('Modules imported successfully.')
