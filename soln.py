import numpy as np 
import pandas as pd 
import seaborn as sns
import os
import tkinter as tk
import matplotlib.pyplot as plt


rankings = pd.read_csv('fifa_ranking.csv')

rankings = rankings.loc[:,['rank', 'country_full', 'country_abrv', 'cur_year_avg_weighted', 'rank_date', 
                           'two_year_ago_weighted', 'three_year_ago_weighted']]
rankings.country_full.replace("^IR Iran*", "Iran", regex=True, inplace=True)
rankings['weighted_points'] =  rankings['cur_year_avg_weighted'] + rankings['two_year_ago_weighted'] + rankings['three_year_ago_weighted']
rankings['rank_date'] = pd.to_datetime(rankings['rank_date'])
rankings.head()




matches = pd.read_csv("results.csv")
matches =  matches.replace({'Germany DR': 'Germany', 'China': 'China PR'})
matches['date'] = pd.to_datetime(matches['date'])
matches.head()


world_cup = pd.read_csv("world_cup.csv")
world_cup = world_cup.loc[:, ['Team', 'Group', 'First match \nagainst', 'Second match\n against', 'Third match\n against']]
world_cup = world_cup.dropna(how='all')
world_cup = world_cup.replace({"IRAN": "Iran", 
                               "Costarica": "Costa Rica", 
                               "Porugal": "Portugal", 
                               "Columbia": "Colombia", 
                               "Korea" : "Korea Republic"})
world_cup = world_cup.set_index('Team')
world_cup.head()


#** Feature extraction**

rankings = rankings.set_index(['rank_date'])\
                    .groupby(['country_full'],group_keys = False)\
                    .resample('D').first()\
                    .fillna(method='ffill')\
                    .reset_index()
rankings.head()




matches = matches.merge(rankings,
                         left_on=['date', 'home_team'],
                         right_on=['rank_date', 'country_full'])
# matches.head()

matches = matches.merge(rankings, 
                        left_on=['date', 'away_team'], 
                        right_on=['rank_date', 'country_full'], 
                        suffixes=('_home', '_away'))
print(matches.head())



fig, ax = plt.subplots()
fig.set_size_inches(20, 20)
corr1 = matches.corr()
corr1
sns.heatmap(corr1,annot=True)
plt.show()

# feature generation
matches['rank_difference'] = matches['rank_home'] - matches['rank_away']
matches['average_rank'] = (matches['rank_home'] + matches['rank_away'])/2
matches['point_difference'] = matches['weighted_points_home'] - matches['weighted_points_away']
matches['score_difference'] = matches['home_score'] - matches['away_score']
matches['is_won'] = matches['score_difference'] > 0 
matches['is_stake'] = matches['tournament'] != 'Friendly'


max_rest = 30
matches['rest_days'] = matches.groupby('home_team').diff()['date'].dt.days.clip(0,max_rest).fillna(max_rest)


matches['wc_participant'] = matches['home_team'] * matches['home_team'].isin(world_cup.index.tolist())
matches['wc_participant'] = matches['wc_participant'].replace({'':'Other'})
matches = matches.join(pd.get_dummies(matches['wc_participant']))





import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
matches = matches.reindex(
       np.random.permutation(matches.index))


def preprocess_features(matches):
    
    selected_features = matches[["average_rank", "rank_difference", "point_difference", "is_stake"]]
    processed_features = selected_features.copy()
    return processed_features

def preprocess_targets(matches):
    output_targets = pd.DataFrame()

    output_targets["is_won"] = matches['is_won']
    return output_targets



training_examples = preprocess_features(matches.head(10900))
training_targets = preprocess_targets(matches.head(10900))


validation_examples = preprocess_features(matches.tail(7267))
validation_targets = preprocess_targets(matches.tail(7267))

Complete_Data_training = preprocess_features(matches)
Complete_Data_Validation = preprocess_targets(matches)



def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):

    
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
     
    ds = Dataset.from_tensor_slices((features,targets)) 
    ds = ds.batch(batch_size).repeat(num_epochs)
    

    if shuffle:
        ds = ds.shuffle(10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_nn_classification_model(
    my_optimizer,
    steps,
    batch_size,
    hidden_units,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
    periods = 10
    steps_per_period = steps / periods

    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 3.0)
    dnn_classifier = tf.estimator.DNNClassifier(
      feature_columns=construct_feature_columns(training_examples),
      hidden_units=hidden_units,
      optimizer=my_optimizer
  )

    training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets["is_won"], 
                                          batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets["is_won"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets["is_won"], 
                                                    num_epochs=1, 
                                                    shuffle=False)


    print("Training model...")
    print("LogLoss (on training data):")
    training_log_losses = []
    validation_log_losses = []
    for period in range (0, periods):

        dnn_classifier.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )

        training_probabilities = dnn_classifier.predict(input_fn=predict_training_input_fn)
        training_probabilities = np.array([item['probabilities'] for item in training_probabilities])
    
        validation_probabilities = dnn_classifier.predict(input_fn=predict_validation_input_fn)
        validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])
    
        training_log_loss = metrics.log_loss(training_targets, training_probabilities)
        validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)

        print("  period %02d : %0.2f" % (period, training_log_loss))
    
        training_log_losses.append(training_log_loss)
        validation_log_losses.append(validation_log_loss)
    print("Model training finished.")

    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.tight_layout()
    plt.plot(training_log_losses, label="training")
    plt.plot(validation_log_losses, label="validation")
    plt.legend()
    plt.show()

    return dnn_classifier


linear_classifier = train_nn_classification_model(
    my_optimizer=tf.train.AdagradOptimizer(learning_rate=0.07),
    steps=3000,
    batch_size=2000,
    hidden_units=[5, 5,6,5],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)



predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                  validation_targets["is_won"], 
                                                  num_epochs=1, 
                                                  shuffle=False)


validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)

validation_probabilities = np.array([item['probabilities'][1] for item in validation_probabilities])

false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
    validation_targets, validation_probabilities)
plt.plot(false_positive_rate, true_positive_rate, label="our model")
plt.plot([0, 1], [0, 1], label="random classifier")
_ = plt.legend(loc=2)
plt.show()


evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)

print("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
print("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])



def train_complete_model(my_optimizer,
    steps,
    batch_size,
    hidden_units,
    Complete_Data_training,
    Complete_Data_Validation) :
    
    periods = 10
    steps_per_period = steps / periods

    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 3.0)
    dnn_classifier = tf.estimator.DNNClassifier(
      feature_columns=construct_feature_columns(Complete_Data_training),
      hidden_units=hidden_units,
      optimizer=my_optimizer
  )
    
    Complete_training_input_fn = lambda: my_input_fn(Complete_Data_training, 
                                          Complete_Data_Validation["is_won"], 
                                          batch_size=batch_size)
    predict_Complete_training_input_fn = lambda: my_input_fn(Complete_Data_training, 
                                                  Complete_Data_Validation["is_won"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
    
    print("Training model...")
    print("LogLoss (on training data):")
    training_log_losses = []
    
    for period in range (0, periods):
   
        dnn_classifier.train(
        input_fn=Complete_training_input_fn,
        steps=steps_per_period
    )
        
        Complete_training_probabilities = dnn_classifier.predict(input_fn=predict_Complete_training_input_fn)
        Complete_training_probabilities = np.array([item['probabilities'] for item in Complete_training_probabilities])
    
        
        training_log_loss = metrics.log_loss(Complete_Data_Validation, Complete_training_probabilities)

        print("  period %02d : %0.2f" % (period, training_log_loss))

        training_log_losses.append(training_log_loss)
      
    print("Model training finished.")
      
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.tight_layout()
    plt.plot(training_log_losses, label="training")

    plt.legend()
    plt.show()

    return dnn_classifier
    


linear_classifier = train_complete_model(
    my_optimizer=tf.train.AdagradOptimizer(learning_rate=0.07),
    steps=3000,
    batch_size=2000,
    hidden_units=[5, 5,6,5],
    Complete_Data_training=Complete_Data_training,
    Complete_Data_Validation=Complete_Data_Validation)



margin = 0.05


world_cup_rankings = rankings.loc[(rankings['rank_date'] == rankings['rank_date'].max()) & 
                                    rankings['country_full'].isin(world_cup.index.unique())]
world_cup_rankings = world_cup_rankings.set_index(['country_full'])


world_cup_rankings.head()


from itertools import combinations

opponents = ['First match \nagainst', 'Second match\n against', 'Third match\n against']

world_cup['points'] = 0
world_cup['total_prob'] = 0
grp1=[]
combi1=[]
win1=[]
for group in set(world_cup['Group']):
    print('___Starting group {}:___'.format(group))
    grp=str('___Starting group {}:___'.format(group))
    grp1.append(grp)
    for home, away in combinations(world_cup.query('Group =="{}"'.format(group)).index, 2):
        print("{} vs. {}: ".format(home, away), end='')
        combi=str("{} vs. {}: ".format(home, away))
        combi1.append(combi)
        row = pd.DataFrame(np.array([[np.nan, np.nan, np.nan, True]]), columns=validation_examples.columns)
        home_rank = world_cup_rankings.loc[home, 'rank']
        home_points = world_cup_rankings.loc[home, 'weighted_points']
        opp_rank = world_cup_rankings.loc[away, 'rank']
        opp_points = world_cup_rankings.loc[away, 'weighted_points']
        row['average_rank'] = (home_rank + opp_rank) / 2
        row['rank_difference'] = home_rank - opp_rank
        row['point_difference'] = home_points - opp_points
        row['is_won'] =np.nan
        predict_validation_input_fn1 = lambda: my_input_fn(row, 
                                                  row["is_won"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
        validation_probabilities1 = linear_classifier.predict(input_fn=predict_validation_input_fn1)
        
        validation_probabilities1 = np.array([item['probabilities'][1] for item in validation_probabilities1])
        
        home_win_prob = validation_probabilities1[0]
        world_cup.loc[home, 'total_prob'] += home_win_prob
        world_cup.loc[away, 'total_prob'] += 1-home_win_prob
        
        points = 0
        if home_win_prob <= 0.5 - margin:
            print("{} wins with {:.2f}".format(away, 1-home_win_prob))
            win=str("{} wins with {:.2f}".format(away, 1-home_win_prob))
            win1.append(win)
            world_cup.loc[away, 'points'] += 3
        if home_win_prob > 0.5 - margin:
            points = 1
        if home_win_prob >= 0.5 + margin:
            points = 3
            world_cup.loc[home, 'points'] += 3
            print("{} wins with {:.2f}".format(home, home_win_prob))
            win=str("{} wins with {:.2f}".format(home, home_win_prob))
            win1.append(win)
        if points == 1:
            print("Draw")
            win=str("Draw")
            win1.append(win)
            world_cup.loc[home, 'points'] += 1
            world_cup.loc[away, 'points'] += 1


pairing = [0,3,4,7,8,11,12,15,1,2,5,6,9,10,13,14]

grp2=[]
combi2=[]
win2=[]

world_cup = world_cup.sort_values(by=['Group', 'points', 'total_prob'], ascending=False).reset_index()
next_round_wc = world_cup.groupby('Group').nth([0, 1]) # select the top 2
next_round_wc = next_round_wc.reset_index()
next_round_wc = next_round_wc.loc[pairing]
next_round_wc = next_round_wc.set_index('Team')

finals = ['round_of_16', 'quarterfinal', 'semifinal', 'final']

labels = list()
odds = list()
pred=[]
for f in finals:
    print("___Starting of the {}___".format(f))
    grp=str("___Starting of the {}___".format(f))
    grp2.append(grp)

    iterations = int(len(next_round_wc) / 2)
    winners = []

    for i in range(iterations):
        home = next_round_wc.index[i*2]
        away = next_round_wc.index[i*2+1]
        print("{} vs. {}: ".format(home,
                                   away), 
                                   end='')
        combi=str("{} vs. {}: ".format(home,away))
        combi2.append(combi)

        row = pd.DataFrame(np.array([[np.nan, np.nan, np.nan, True]]), columns=validation_examples.columns)
        home_rank = world_cup_rankings.loc[home, 'rank']
        home_points = world_cup_rankings.loc[home, 'weighted_points']
        opp_rank = world_cup_rankings.loc[away, 'rank']
        opp_points = world_cup_rankings.loc[away, 'weighted_points']
        row['average_rank'] = (home_rank + opp_rank) / 2
        row['rank_difference'] = home_rank - opp_rank
        row['point_difference'] = home_points - opp_points
        row['is_won'] =np.nan
        predict_validation_input_fn1 = lambda: my_input_fn(row, 
                                                  row["is_won"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
        validation_probabilities1 = linear_classifier.predict(input_fn=predict_validation_input_fn1)
        
        validation_probabilities1 = np.array([item['probabilities'][1] for item in validation_probabilities1])
        
        home_win_prob = validation_probabilities1[0]

        
        if home_win_prob <= 0.5:
            print("{0} wins with probability {1:.2f}".format(away, 1-home_win_prob))
            win=str("{0} wins with probability {1:.2f}".format(away, 1-home_win_prob))
            win2.append(win)
            winners.append(away)
        else:
            print("{0} wins with probability {1:.2f}".format(home, home_win_prob))
            win=str("{0} wins with probability {1:.2f}".format(home, home_win_prob))
            win2.append(win)
            winners.append(home)

        labels.append("{}({:.2f}) vs. {}({:.2f})".format(world_cup_rankings.loc[home, 'country_abrv'], 
                                                        1/home_win_prob, 
                                                        world_cup_rankings.loc[away, 'country_abrv'], 
                                                        1/(1-home_win_prob)))
        odds.append([home_win_prob, 1-home_win_prob])
                
    next_round_wc = next_round_wc.loc[winners]
    pred.append(next_round_wc)
    print("\n")





#GUI OUTPUT
outF = open("myOutFile.txt", "w")

outF.write(grp1[0])
outF.write("\n")
for g in range (0,6):
        outF.write(combi1[g] + win1[g])
        outF.write("\n")
outF.write("==========================================================")
outF.write("\n")


outF.write(grp1[1])
outF.write("\n")
for g in range (6,12):
        outF.write(combi1[g] + win1[g])
        outF.write("\n")
outF.write("==========================================================")
outF.write("\n")


outF.write(grp1[2])
outF.write("\n")
for g in range (12,18):
        outF.write(combi1[g] + win1[g])
        outF.write("\n")


outF.write("==========================================================")
outF.write("\n")

outF.write(grp1[3])
outF.write("\n")
for g in range (18,24):
        outF.write(combi1[g] + win1[g])
        outF.write("\n")

outF.write("==========================================================")
outF.write("\n")


outF.write(grp1[4])
outF.write("\n")
for g in range (24,30):
        outF.write(combi1[g] + win1[g])
        outF.write("\n")

outF.write("==========================================================")
outF.write("\n")


outF.write(grp1[5])
outF.write("\n")
for g in range (30,36):
        outF.write(combi1[g] + win1[g])
        outF.write("\n")

outF.write("==========================================================")
outF.write("\n")


outF.write(grp1[6])
outF.write("\n")
for g in range (36,42):
        outF.write(combi1[g] + win1[g])
        outF.write("\n")

outF.write("==========================================================")
outF.write("\n")


outF.write(grp1[7])
outF.write("\n")
for g in range (42,48):
        outF.write(combi1[g] + win1[g])
        outF.write("\n")

outF.write("==========================================================")
outF.write("\n")


outF.write(grp2[0])
outF.write("\n")
for g in range (0,8):
        outF.write(combi2[g] + win2[g])
        outF.write("\n")

outF.write("==========================================================")
outF.write("\n")


outF.write(grp2[1])
outF.write("\n")
for g in range (8,12):
        outF.write(combi2[g] + win2[g])
        outF.write("\n")

outF.write("==========================================================")
outF.write("\n")


outF.write(grp2[2])
outF.write("\n")
for g in range (12,14):
        outF.write(combi2[g] + win2[g])
        outF.write("\n")
outF.write("==========================================================")
outF.write("\n")

outF.write(grp2[3])
outF.write("\n")
outF.write(combi2[14] + win2[14])
outF.write("\n")

outF.write("==========================================================")

from tkinter import *
from PIL import Image, ImageTk
from tkinter import messagebox




root = Tk()  # Main window 
f = Frame(root)
frame1 = Frame(root)
frame2 = Frame(root)
frame3 = Frame(root)
root.title("Foootball Winner Prediction")
root.geometry("1080x720")

canvas = Canvas(width=1080, height=250)
canvas.pack()
filename=('sports.png')
load = Image.open(filename)
load = load.resize((1800, 250), Image.ANTIALIAS)
render = ImageTk.PhotoImage(load)
img = Label(image=render)
img.image = render
#photo = PhotoImage(file='landscape.png')
load = Image.open(filename)
img.place(x=1, y=1)
#canvas.create_image(-80, -80, image=img, anchor=NW)

root.configure(background='#FCFCE5')
scrollbar = Scrollbar(root)
scrollbar.pack(side=RIGHT, fill=Y)

firstname = StringVar()  # Declaration of all variables
lastname = StringVar()
id = StringVar()
dept = StringVar()
designation = StringVar()
remove_firstname = StringVar()
remove_lastname = StringVar()
searchfirstname = StringVar()
searchlastname = StringVar()
sheet_data = []
row_data = []



def add_entries():  # to append all data and add entries on click the button
    a = " "
    f = firstname.get()
    f1 = f.lower()
    l = lastname.get()
    l1 = l.lower()
    d = dept.get()
    d1 = d.lower()
    de = designation.get()
    de1 = de.lower()
    list1 = list(a)
    list1.append(f1)
    list1.append(l1)
    list1.append(d1)
    list1.append(de1)


def click( ):
    filename='myOutFile.txt'
    #File1 = tkFileDialog.askopenfilename()
    File2 = open(filename, "r")
    e1.delete( END)
    e1.insert('1.0',File2.read())




def clear_all():  # for clearing the entry widgets
    frame1.pack_forget()
    frame2.pack_forget()
    frame3.pack_forget()




label1 = Label(root, text="Foootball Winner Prediction")
label1.config(font=('Italic', 18, 'bold'), justify=CENTER, background="#fdfe02", fg="#133e7c", anchor="center")
label1.pack(fill=X)


frame2.pack_forget()
frame3.pack_forget()



e1 = Text(frame1,height=15, width=70)
e1.grid(row=1, column=2, padx=10,pady=10)




button5 = Button(frame1, text="Predict",command=click)
button5.grid(row=9, column=2, pady=10,padx=10)



# textbox = tk.Text(frame2, font=10,width="15",height=70)
# textbox.grid(row=3, column=1)



frame1.configure(background="#030056")
frame1.pack(pady=10)

frame2.configure(background="#030056")
frame2.pack(pady=10)

root.mainloop()

















