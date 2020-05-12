
# """
# Andrew Beck
# A01009492
# Final Project
# """


# """
# !!!!! Warning, this takes a bit to run, not broken just slow, 
# the dataset is 200mb !!!!!!

# Currently most of the cleaning steps are commented out.
# I have provided a cleaned and feature engineered file to avoid
# the 3 hours processing time this code currently takes on my machine. 

# Citations at the bottom of the code.

# Problem:

# Terrorists groups often are unknown at the outset of an attack or 
# falsely claim responsibility for an attack. The ability to predict or verify
# with statistics which terrorist group commited an attack would be useful. 
# Whether or not this prediction is possible with the dataset available
# remains to be seen and will be explored in this project.




# Some notes about this dataset:

# 'The Global Terrorism Database (GTD) is an open-source database including 
# information on terrorist events around the world from 1970 through 2017 
# (with annual updates planned for the future). 
# Unlike many other event databases, the GTD includes systematic data on 
# domestic as well as international terrorist incidents that have occurred 
# during this time period and now includes more than 180,000 cases.' 

# Definition of terrorism:

# "The threatened or actual use of illegal force and violence by a non-state 
# actor to attain a political, economic, religious, or social goal through 
# fear, coercion, or intimidation."
# - 
# https://www.start.umd.edu/gtd/

# The dataset currently cuts off at 2017 as START lost their funding 
# in mid-2018 due to budget cuts. 


# The datasets definition of terrorism is rather, broad. 
# For example state actors such as El Salvador's
# Farabundo Martí National Liberation Front (FMNL) are included in the dataset
# despite being civil war participants.

# The Data set includes 183 features and is well maintained and clean for the
# most part.


# """

import pandas as pd 
import numpy  as np 
#Seaborn import keeps randomly breaking so commented out for now.
#import seaborn as sns 

import matplotlib.pyplot as plt
from scipy import stats
from sklearn.utils.multiclass import unique_labels
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import keras.utils
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder


"""
Using read_excel here instead of read_csv due to the
large amount of columns that would require me to specify the data type to 
allow pandas to load the csv. This is really inconvenient so I suggest using 
the excel file that is available directly from START instead of the Kaggle CSV
That or running the file from directly from databricks may be wise. 
"""

df = None



df = pd.DataFrame(pd.read_excel('C:/Users/Andrew/dev/data_analytics/'
    'globalterrorismdb_0718dist.xlsx')) 



#alternatively instead hardcoding use below prompt
"""
while df is None :
    try:
        df = pd.DataFrame(pd.read_excel(input('Please enter full csv file path: ')))

    except Exception as e:
        print('Invalid CSV path')
"""

"""

Cleaning data below.

Removing uncessary or sparse columns 
these columns are lightly populated or are redundant
or have paragraph text or notes about the record.
For example, summary is essentially a short paragraph that describes
and is not populated on most records.
Some may be useful but I was already reaching the my computers
hardware limits with the amount of features I was engineering

"""
dropcol = ['eventid','approxdate','resolution','location','summary',
'alternative','alternative_txt','attacktype2','attacktype2_txt',
'attacktype3','attacktype3_txt','gsubname','gname2',
'gsubname2','gname3','gsubname3','motive','guncertain2','guncertain3',
'claim2','claimmode2','claimmode','country','region',
'claim3','claimmode3','claimmode3_txt','compclaim','weaptype1',
'weapsubtype1','weapsubtype1_txt','weaptype2','weaptype2_txt',
'weapsubtype2','weapsubtype2_txt','weaptype3','weaptype3_txt','weapsubtype3',
'weapsubtype3_txt','weaptype4','weaptype4_txt','weapsubtype4',
'weapsubtype4_txt','weapdetail','nkillus','nwoundus','latitude',
'longitude','specificity','vicinity','divert','kidhijcountry',
'ransomamtus','ransompaidus','propcomment','nhostkidus',
'hostkidoutcome','addnotes','scite1','scite2','scite3',
'dbsource','target1','target2','target3','related','hostkidoutcome_txt',
'targtype2','targtype2_txt','targsubtype2','targsubtype2_txt','corp2',   
'target2','natlty2', 'targtype3','targtype3_txt','targsubtype3',
'targsubtype3_txt','corp3','target3','natlty1', 'natlty3', 'natlty3_txt',
'attacktype1','targtype1','targsubtype1','provstate','city','targsubtype1_txt',
'corp1','natlty2_txt','propextent','propextent_txt',
'claimmode_txt','claimmode2_txt','ransomnote'
]


df.drop(dropcol, axis=1, inplace=True)
# 
# 
#Remove any unknown groups, we're not interested in 
#predicting unknown attacks
#This also skews the data as it makes up roughly half the dataset.
#
df = df[df.gname != 'Unknown']
# 
#Remove any attacks prior to 2007 to modernize the dataset somewhat.
#This means the data will span the last decade.
#This is important becauase if we want our model to be useful having it 
#represent modern groups which exist is important.
# 
df = df[df.iyear >= 2007]


#Update values of related groups to flatten classes
df.loc[df.gname.str.contains('Al-Qaida'),'gname'] = 'Al-Qaida'
df.loc[df.gname.str.contains('Maoist'),'gname'] = 'Maoists'
df.loc[df.gname.str.contains('Baloch'),'gname'] = 'Baloch Liberation'
df.loc[df.gname.str.contains('Islamic State'),'gname'] = 'ISIS / ISIL'
df.loc[(df.gname.str.contains('Islamist extremists')) | 
(df.gname.str.contains('Algerian Islamic Extremists')) |
(df.gname.str.contains('Muslim extremists')) |
(df.gname.str.contains('Jihadi-inspired extremists')) |
(df.gname.str.contains('Jihadi-inspired extremists')),'gname'] = 'Islamic'
'Extremists'
df.loc[df.gname.str.contains('Palestin'),'gname'] = 'Palestinian Extremists'

#Remove any groups that have fewer than 200 incidents
df = df[df.groupby('gname')['gname'].transform('size') > 199]

#The dataset codes 'Unknown' in binary columns as -99 or  -9 need to fix this
#For this project I decided that 'Unknown' binary values were akin to 0

df['nperps'].replace(-99, 0, inplace = True)
df['nperpcap'].replace(-99, 0, inplace = True)
df['nhostkid'].replace(-99, 0, inplace = True)
df['nhours'].replace(-99, 0, inplace = True)
df['ransomamt'].replace(-99, 0, inplace = True)
df['ransompaid'].replace(-99, 0, inplace = True)
df['nreleased'].replace(-99, 0, inplace = True)
df['ndays'].replace(-99, 0, inplace = True)
df['propvalue'].replace(-99, 0, inplace = True)
df['INT_LOG'].replace(-9, 0, inplace = True)
df['INT_IDEO'].replace(-9, 0, inplace = True)
df['INT_MISC'].replace(-9, 0, inplace = True)
df['INT_ANY'].replace(-9, 0, inplace = True)
df['doubtterr'].replace(-9, 0, inplace = True)
df['property'].replace(-9, 0, inplace = True)
df['ransom'].replace(-9, 0, inplace = True)
df['claimed'].replace(-9, 0, inplace = True)
df['ishostkid'].replace(-9, 0, inplace = True)
df['nhours'].replace(-9, 0, inplace = True)

df.to_excel('C:/Users/Andrew/dev/data_analytics/clean.xlsx')


#At this point needed to do a small bit of excel work to handle 
#some NaN values that random forest doesn't like. Essentially
#replaced all blank values with 0s. 

#alternatively use this lambda below


df.apply(lambda x: x = 0 if(x != x OR x == '' OR x == None))

"""
Some quick visualization of the correlations
between the remaining features.
Note this visualization takes place prior to feature
engineering

"""
Var_Corr = df.corr()

plot1 = sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, 
yticklabels=Var_Corr.columns, annot=True)


"""
This figure is availabe under fig1.png in the project folder

There are some interesting and likely 
explainable correalations here:

* The nationality of the attackers and the country an 
attack takes place in are closely correalated (0.6)

* the year attack takes place correalates with the region the attack takes
place in, likely because regions like Europe go through periods of unrest
and then largely settle down, for example 'The Troubles' in Ireland

* The length of an attack in hours is negatively correlated to the year
the attack takes place in. Are attacks getting shorter?


"""




"""
Feature Extraction / Engineering:

I will need dummy binary features to handle all the text in this dataset
To do so im going to use pandas dummy feature. This feature allows
pandas to create binary features based on distinct text values
I'm going to get a lot of them, so this is likely going to slow the 
processing of the code down and greatly increase the runtime of the script. 
This is why I've decided to truncate the 
table so much above. 

Having city,prov/state and country made this code take over
an 3 hours to run so I cut them from the dataset even if they
would likely be pertinent to the model. Leaving only region as
a geographic indicator may cause some issues for the model

"""


textcol = ['region_txt','attacktype1_txt','targtype1_txt',
        'natlty1_txt','weaptype1_txt']

df = pd.concat([df, pd.get_dummies(df[textcol]).rename(columns = 
"{}_binary".format)
    ], 
    axis = 1)

df.drop(textcol, axis=1, inplace=True)



"""

Terrorist group names will represent the classes that we're trying to predict. 
Lets take a look at how much of the dataset each group represents and how 
many distinct classes there are. 



"""


vc = df['gname'].value_counts()
print(vc)
plot2 = sns.catplot(y = df['gname'], kind = "count", data = df, order = 
    df.gname.value_counts().index)




"""
For my project I decided to use a Random Forest classifier.
I chose RF because we had not used it in class and I wanted
to work on a classification problem. 

Because some of my 'trees' show high correlation to a specific
group (such as region), I think RF is a good use for this problem. 


The above plot shows that the data is unbalanced.
I may have to deal with this later. 

Some groups make up a heavy weight of the dataset while
others are barely present in comparison. 

Now I must split my data into training data and test data.

"""

x_cols = [x for x in df.columns if x != 'gname']

X = df[x_cols]
y = df['gname']
xtrain, xtest, ytrain, ytest  = train_test_split(X, y,
        train_size = 0.8, random_state = 21)
class_names = df['gname'].unique().tolist()

clf = RandomForestClassifier()
clf.fit(xtrain, ytrain)

predictions = clf.predict(xtest)

for i in range(0,200):
    print("Actual outcome :: {} and Predicted outcome :: {}".format(list(ytest)
        [i], predictions[i]))





"""
Code for plotting confusion matrix plot taken from here
https://scikit-learn.org/stable/auto_examples/model_selection/
plot_confusion_matrix.html

"""

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

plot_confusion_matrix(ytest, predictions, classes= class_names, normalize=True,
                      title='Normalized confusion matrix')





print("Train Accuracy :: ", accuracy_score(ytrain, clf.predict(xtrain)))
print("Test Accuracy  :: ", accuracy_score(ytest, predictions))

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[`f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()




"""
This looks pretty good

Train accuracy is: .9952

Test accuracy is: .9282

The model is roughly 92% accurate
Likely cause of miss-classification appears to be statistically
unlikely groups that share regions with more prominent groups

That or it is caused by an accuracy paradox, perhapst the imbalance
of the dataset is causing issues. 
To test against this I will create a second model with a stratified split

"""



xtrain, xtest, ytrain, ytest  = train_test_split(X, y,
        train_size = 0.8, random_state = 21, stratify = y)
class_names = df['gname'].unique().tolist()

clf = RandomForestClassifier()
clf.fit(xtrain, ytrain)

predictions = clf.predict(xtest)

for i in range(0,200):
    print("Actual outcome :: {} and Predicted outcome :: {}".format(list(ytest)
        [i], predictions[i]))

plt.clf()

plot_confusion_matrix(ytest, predictions, classes= class_names, normalize=True,
                      title='Normalized confusion matrix')


plt.show()


print("Train Accuracy :: ", accuracy_score(ytrain, clf.predict(xtrain)))
print("Test Accuracy  :: ", accuracy_score(ytest, predictions))


"""
The stratified model performace is essentially the same
Accuracy = .9243

My belief is that the imbalance in the data does not
create too much of a problem for the random forest classifier
But unfortunately it isn't perfect. Likely could be made better if classes
were further subdivided. 


The confusion matrix shows there are some issues with larger classes
such as ISIS. Breaking these classes up may allieviate some of the
accuracy issues. 
"""



"""
I had some additional time and decided to play with Keras

I followed a few tutorials to build a classifier below but
it currrently isn't working so commented out. 

"""
# xtrain, xtest, ytrain, ytest  = train_test_split(X, y,
#         train_size = 0.8, random_state = 21, stratify = y)
# class_names = df['gname'].unique().tolist()

# ytrainbinary = to_categorical(ytrain)
# ytestbinary = to_categorical(ytest) 

# model = Sequential()


# model.add(Dense(100,activation = 'relu',input_dim = len(X.columns),
#     kernal_regularizer = regularizers.l2(0.01)))
# model.add(Dropout(0.3, noise_shape = None, seed = None))


# model.add(Dense(100,activation = 'relu',
#     kernal_regularizer = regularizers.l2(0.01)))
# model.add(Dropout(0.3, noise_shape = None, seed = None))


# model.add(Dense(1, activation = 'sigmoid'))



# model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',
#     metrics = ['accuracy'])

# mode_output = model.fit(xtrain, ytrain, epochs = 500, batch_size  = 20, 
#     verbose = 1, validation_data = (xtest,ytest),)

# print("Train Accuracy :: ", np.mean(model_output.history['acc']))
# print("Test Accuracy  :: ", np.mean(model_output.history['val_acc']))


# """
# References

# Random Forest:
# https://towardsdatascience.com/the-random-forest-algorithm-d457d499ffcd

# https://www.datascience.com/resources/notebooks/random-forest-intro

# Dealing with imbalanced samples:

# https://statistics.berkeley.edu/sites/default/files/tech-reports/666.pdf

# https://stackoverflow.com/
# questions/40565444/balanced-random-forest-in-scikit-learn-python

# https://towardsdatascience.com/
# machine-learning-multiclass-classification-with-imbalanced-data-set-29f6a177c1a

# Confusion matrix plot code

# https://scikit-learn.org/stable/auto_examples/model_selection/
# plot_confusion_matrix.html

# Keras
# Learning a keras classification model
# https://towardsdatascience.com/
# k-as-in-keras-simple-classification-model-a9d2d23d5b5a
# https://www.quora.com/How-do-I-set-an-input-shape-in-Keras


# """