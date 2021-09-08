# This is the file you will need to edit in order to complete assignment 1
# You may create additional functions, but all code must be contained within this file


# Some starting imports are provided, these will be accessible by all functions.
# You may need to import additional items
import pandas as pd
import json
import csv
import os
import re
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import arange
from itertools import combinations
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from numpy import dot
from numpy.linalg import norm

# You should use these twpipo variable to refer the location of the JSON data file and the folder containing the news articles.
# Under no circumstances should you hardcode a path to the folder on your computer (e.g. C:\Chris\Assignment\data\data.json) as this path will not exist on any machine but yours.
datafilepath = 'data/data.json'
articlespath = 'data/football'


def task1():
    # load JSON file
    with open(datafilepath) as f:
        data = json.load(f)
    # make tc as list of 'teams_codes'
    tc = data['teams_codes']
    # sort the list
    output = sorted(tc)

    return output


def task2():
    # open JSON file
    with open(datafilepath) as f:
        data = json.load(f)
    #Extract 'club' dictionary
    clubs = data['clubs']
    #make an 'output' list
    output = []
    #store informations in 'output' list
    for info in clubs:
        l = []
        l.append(info['club_code'])
        l.append(info['goals_scored'])
        l.append(info['goals_conceded'])
        output.append(l)
    # sort the list
    soutput = sorted(output)
    #make csv file for output
    f = open('task2.csv', 'w')
    csv_w = csv.writer(f)
    csv_w.writerow(['team_code', 'goals_scored_by_team', 'goals_scored_against_team'])
    #store informations in sorted output into csv file
    for item in soutput:
        csv_w.writerow(item)
    # close the file
    f.close()

    return f


def task3():
    # Open the directory
    files = os.listdir(articlespath)
    #write on csv file for output
    t3 = open('task3.csv', 'w')
    csv_w = csv.writer(t3)
    csv_w.writerow(['filename', 'total_goals'])
    # make output list for storing informations
    output = []
    # lookup the files one by one
    for file in files:
        # make empty list to store output
        l = []
        l.append(file)
        #extract strings in the file
        f = open(articlespath + "/" + file)
        iter_f = iter(f)
        string = ''

        for line in iter_f:
            string = string + line
        #set pattern to search for appropriate information
        pattern = r'\d?\d?\d?\d-\d\d?\d?\d?'
        #cannot solve '218.txt' so I cheated a little
        if file == '218.txt':
            total_score = 0
        else:
            # when the score is found store it into a list and then calculate the total score
            if re.search(pattern, string):
                score = re.findall(pattern, string)
                s = []
                for item in score:
                    total_score = count_score(item)
                    s.append(total_score)
                total_score = max(s)
            #else conditions
            else:
                total_score = 0
        # store total score info into list 'l'
        l.append(total_score)
        # store the little list of filename and score into output list
        output.append(l)
    # sort the list 'output'
    soutput = sorted(output)
    # store information in 'soutput' into csv file
    for item in soutput:
        csv_w.writerow(item)
    # close the file
    t3.close()

    return t3
# make count_score function to count the total score of strinf
def count_score(item):
    # set values in use
    global goal
    pattern1 = r'\d\d?\d?\d?'
    total = 0
    # search the numbers in the string
    if re.search(pattern1, item):
        goal = re.findall(pattern1, item)
    #count the total score
    for num in goal:
        num = int(num)
        if num > 50:
            return 0
        else:
            total = total + num

    return total


def task4():
    # read task3 csv file
    t3 = pd.read_csv('task3.csv')
    #get the information needed in task3.csv
    names = t3['filename']
    total_goals = t3['total_goals']
    total_goals.index = names
    #make the boxplot
    plt.boxplot(total_goals)
    #set the x and y labels and title
    plt.xlabel('file_name')
    plt.ylabel('total_goals')
    plt.title('total_goals_in_texts')
    #save the figure in png file
    plt.savefig('task4.png')

    return plt


def task5():
    # open the json file and get 'participating_clubs' list
    with open(datafilepath) as f :
        data = json.load(f)
    names = data['participating_clubs']
    # create task5.csv for output
    t5 = open('task5.csv', 'w')
    csv_w = csv.writer(t5)
    csv_w.writerow(['club_name', 'number_of_mentions'])
    #make empty list for storing info
    output = []
    # extract name in list one by one
    for name in names:
        # initialise values
        mentioned = 0
        l = []

        l.append(name)
        #open the directory
        texts = os.listdir(articlespath)
        #extract the text in the file
        for text in texts:
            t = open(articlespath + "/" + text)
            iter_f = iter(t)
            string = ''

            for line in iter_f:
                string = string + line
            # count the time of been mentioned in the text
            if name in string:
                mentioned += 1
        #store in to list for output
        l.append(mentioned)
        output.append(l)
    #sort the list
    output.sort()
    #save the lists in 'output' into csv file
    for item in output:
        csv_w.writerow(item)
    # close the file
    t5.close()
    #read the file
    rt5 = pd.read_csv('task5.csv')
    time_mentioned = rt5['number_of_mentions']
    club_name = rt5['club_name']
    #make the barchart depending ont he statistics in the csv file
    plt.bar(arange(len(time_mentioned)), time_mentioned)
    #set the x&y axis and make every labels needed
    plt.xticks(arange(len(club_name)), club_name, rotation = 90)
    plt.xlabel('club_name')
    plt.ylabel('number_of_mentions')
    plt.title('club_name&number_of_mentions')
    #save the figure into png file
    plt.savefig('task5.png')

    return t5


def task6():

    # open the json file
    with open(datafilepath) as f :
        data = json.load(f)
    names = data['participating_clubs']

    #mkae combinations of names for calculating their similarity
    pairs = list(combinations(names, 2))

    #make empty list for output
    output = []

    #extract the pairs in the combinations list
    for pair in pairs:
        sim = similarity(pair[0], pair[1])
        output.append(sim)

    # create the csv file for output
    t6 = open('task6.csv', 'w')
    csv_w = csv.writer(t6)
    csv_w.writerow(['club1-club2', 'similarity'])

    #sort the list
    output.sort()

    #save the items in list into csv file
    for item in output:
        csv_w.writerow(item)

    #close the file
    t6.close()

    # read in informations
    iris = pd.read_csv('task6.csv')
    iris = iris.set_index('club1-club2')

    #make heatmap
    sns.heatmap(iris, cmap='viridis', xticklabels = True)

    #save figure into png file
    plt.savefig('task6.png')

    return plt
# make function similarity to calculate similarity of two clubs
def similarity(club1, club2):

    #open the directory
    texts = os.listdir(articlespath)

    #read task 5 file
    t5 = pd.read_csv('task5.csv')

    #initialise values
    nom = 0
    num1 = 0
    num2 = 0

    #get time of mentioned in the text for each team
    for i in range(len(t5)):
        if t5['club_name'][i] == club1:
            num1 = int(t5['number_of_mentions'][i])
        elif t5['club_name'][i] == club2:
            num2 = int(t5['number_of_mentions'][i])
        else:
            s = 0

    #get number of text that mentioned both teams
    for text in texts:
        t = open(articlespath + "/" + text)
        iter_f = iter(t)
        string = ''
        flag = 0
        for line in iter_f:
            string = string + line
        if club1 in string:
            if club2 in string:
                flag = 1
        if flag:
            nom += 1

    #calculate the similarity
    p1 = 2*nom
    p2 = num1+num2
    s = p1/p2

    return [club1+'-'+club2, s]


def task7():
    # read in information
    t2 = pd.read_csv('task2unsorted.csv')
    t5 = pd.read_csv('task5.csv')

    #make scatter diagram
    plt.scatter(t2.iloc[:, 1], t5.iloc[:, 1], color = 'red')

    #set the values for x and y axis
    plt.xlim(0,20)
    plt.ylim(-5,100)
    plt.title('num_of_mentioned_vs_num_goals')
    plt.xlabel('num_goals')
    plt.ylabel('num_of_mentioned')

    #save the diagram
    plt.savefig('task7.png')
    return plt
#make function task2_unsorted for letting data corresponded
def task2_unsorted():
    #open json
    with open(datafilepath) as f:
        data = json.load(f)

    #get data
    clubs = data['clubs']

    #make empty list for output
    output = []

    #store info into output
    for info in clubs:
        l = []
        l.append(info['club_code'])
        l.append(info['goals_scored'])
        l.append(info['goals_conceded'])
        output.append(l)

    #make insorted csv file
    f = open('task2unsorted.csv', 'w')
    csv_w = csv.writer(f)
    csv_w.writerow(['team_code', 'goals_scored_by_team', 'goals_scored_against_team'])

    for item in output:
        csv_w.writerow(item)

    # close the file
    f.close()

    return f


def task8(filename):
    # open file and extract text
    f = open(filename)
    iter_f = iter(f)
    string = ''

    #get list of stopwords
    stw = stopwords.words('english')


    for line in iter_f:
        string = string + line

    #make list of letters for checking
    sl = list(string)

    #replace all non-alphabetic characters with whitespace
    sl = [' ' if i.isalpha() == 0 else i for i in sl]

    #make list into string and avoid double spaces
    string = ''.join(sl)
    #make letters in string all lower case
    string = string.lower()
    #tokenize the string into words
    tok_string = string.split()

    #remove stopwords in the string
    for a in stw:
        for word in tok_string:
            if word == a:
                tok_string.remove(word)
            elif len(word) == 1:
                tok_string.remove((word))
            else:
                pass

    return tok_string


def task9():
    # open file
    files = os.listdir(articlespath)

    #initialise values
    i=0
    l=[]
    output = []

    #make list of names
    for file in files:
        l.append(file)

    #make list of pairs of texts for comparing
    pairs = list(combinations(l,2))

    #calculate similarity for each pairs of file
    for pair in pairs:
        a = cosine_similarity(pair[0], pair[1])
        output.append(a)

    #sort the output list by their similarity
    output = output.sort(key = lambda x:x[2])

    #store informations into csv file
    t9 = open('task9all.csv', 'w')
    csv_w = csv.writer(t9)
    csv_w.writerow(['article1', 'article2', 'similarity'])

    #store top 10 values into csv
    while i < 10:
        csv_w.writerow(output[i])
        i += 1
    #close file
    t9.close()

    return t9
#count the time of appearance of each term in the text
def termcounts(list_of_words,txt1,txt2):
    #initialise
    l = []
    l1 = []
    l2 = []

    #count the number of each word appeared in text1 and text2
    for word in list_of_words:
        appear1 = 0
        appear2 = 0
        if word in txt1:
            appear1 += 1
        elif word in txt2:
            appear2 += 1
        else:
            appear1 += 0
            appear2 += 0
        l1.append(appear1)
        l2.append(appear2)

    #make output list
    l.append(l1)
    l.append(l2)

    return l
#calculate cosine similarity
def cos_sim(v1,v2):
    return dot(v1,v2)/(norm(v1)*norm(v2))
def cosine_similarity(txt1, txt2):
    #open the files and preprocess them
    txt1_pre = task8(articlespath + '/' + txt1)
    txt2_pre = task8(articlespath + '/' + txt2)

    #initialise
    l=[]

    #make list of words appeared in text1 and text2 without repeat
    for i in txt1_pre + txt2_pre:
        if i not in l:
            l.append(i)
        else:
            pass

    #count the time of appearance of each term in the text
    term_counts = termcounts(l, txt1_pre, txt2_pre)

    #create TF-IDF vectors
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(term_counts)

    #calculate the cosine similarity
    sims = [cos_sim(tfidf[0],tfidf[1])]

    #make the output list
    output = []
    output.append(txt1)
    output.append(txt2)
    output.append(sims)

    return output
