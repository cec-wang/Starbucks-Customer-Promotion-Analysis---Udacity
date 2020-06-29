# Starbucks-Customer-Promotion-Analysis---Udacity
This projected is created as part of the fulfillment for Udacity Data Science Nanodegree. Here I take data on customer demographic, promotions and customer purchasing history and try to recommend the best promotion for the customer. 

## Motivation: 
I take experimental data collected over a period of one month by Starbucks on their customer's purchasing history, and discover what are the groups in there and what are the offers that really excite people. The purpose of the project is to discover the best offer for customers at Starbucks. 

## Instruction:
* link to the write-up: https://www.linkedin.com/pulse/draft/AgEzrx6vxj78OAAAAXL-iBgfFIfcpCAnLfQOcEIHP2afw-TD2_PMBQdStZ9r71ryDfJf8ec
* Make sure you import starbucks into Starbucks_Capstone_notebook.ipynb by `import starbucks`

## Data: 
To run the program successfully, you will need three sets of data: 

**portfolio.json**
* id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)

**profile.json**
* age (int) - age of the customer
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

**transcript.json**

* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record

## Files
*Starbucks.py:*
Python file containing functions to be used for data cleaning and analysis. Need to be imported before used. 

*Starbucks_Capstone_notebook.ipynb:*
Main body of the project. Here you run the functions and see the results

*Starbucks_Capstone_notebook.html:*
An html format of the above file

*data foler*:
 * *portfolio.JSON*: JSON file containing offers information
 
 * *profile.JSON*: JSON file containing members information
 
 * *transcript.JSON*: JSON file containing all member activities

## Functions
*Starbucks.py:*
* `test_print()` : print out a line telling you the functions are imported successfully
* `channel_separation(data = portfolio)`: Cleans portfolio data and separate categorial columns into separate columms.
* `portfolio_data_cleaning(portfolio = portfolio)`: cleans the portfolio data
* `profile_clean(profile = profile)`: cleans the profile data
* `generate_transform_transcript(transcript)`: generates the transformed transcript
* `transcript_clean(transcript = transcript_2, portfolio = portfolio)`: cleans the transcript data
* `event_grouped_count(event, groupby, transcript = transcript)`: filter the transcript by event, then group by groupby
* `generate_duration_dict(portfolio)`: generates an diction with offer id as keys and offer duration as values.


## Libraries
* pandas
* MatplotLib
* Numpy
* Seaborn
* Scikitlearn
* Progressbar
* math
* json

## Analysis Results
1. *Analysis*: Data exploration and visualization
    
    there does not seem to be a significant difference among different demographic characters across offer attributes.  People with higher salary seems to have a slight preference on offer than with no offers. Older people spend more than younger people, but are more likely to spend with offers. This might indicate that it can be difficult to predict an individual's preference based on his/her demographic characters. There are two possibilities: 1. this is how it is. Preference towards different offer attributes will depend on some other characters than demographic information. 2. Data not collected or processed well enough to show statistical significance. I will discuss this in detail in the Improvement section. 
2. *Clustering*:
    1. Density-based clustering
            * On offer attributes and demographic data combined
              There is a weak indication that 1. the more wealthy people would be more likely to spend with an offer. 2. Male are more likely to spend without an offer.
            * On offer attributes only, but later find demographic characters based on the clusters
              Distance between samples too homogeneous to be clustered. 
    2. k-means clustering
            * On offer attributes only.
             Positive link between age, salary and preference towards offers.
    3. Agglomerative Clustering
            * On offer attributes only.
            Distance between samples too homogeneous to be clustered. 
3. *Regression*:
    1. K - Neighbors Regression
    2. Extra Tree Regression
    
    The results show that k nearest neighbors is better for regression, since it has both lower MSE and higher R-squared. Best MSE achieved is 313076, and highest R-squared achieved is 0.17, using 60 neighbors.
    
## Reference
1. [DBSCAN Python Example: The Optimal Value For Epsilon](EPShttps://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc)
2. [scikit-learn library](https://scikit-learn.org/stable/index.html)
3. [Preprocessing with sklearn: a complete and comprehensive guide](https://towardsdatascience.com/preprocessing-with-sklearn-a-complete-and-comprehensive-guide-670cb98fcfb9)
4. [matplotlib library](https://matplotlib.org/3.1.0/index.html)
5. [Udacity](https://classroom.udacity.com/me)

## Acknowledgement
This project is fulfilled as a part of Udacity Data Science Nanodegree.
