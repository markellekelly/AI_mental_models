import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ====================== relevant data/info ======================================

sid_to_topic = {"1":1, "8":2, "15":3, "20":4} # SID 1: art, 8: video games, 15: cities, 20: math

AI_performances_good = {'1': 10, '2': 10, '3': 9, '4': 8, '5': 10, '6': 11, '7': 6, '8': 8,
                        '9': 6, '10': 9, '11': 9, '12': 10, '13': 11, '14': 10, '15': 11, '16': 11}

AI_performances_bad = {'1': 2, '2': 3, '3': 5, '4': 4, '5': 8, '6': 6, '7': 4, '8': 7, '9': 3,
                       '10': 3, '11': 5, '12': 4, '13': 5, '14': 9, '15': 7, '16': 9}

human_performances_good = [{'1': 10, '2': 9, '3': 5, '4': 5, '5': 10, '6': 11, '7': 8, '8': 9, '9': 6, '10': 8, '11': 8, '12': 5, '13': 12, '14': 10, '15': 12, '16': 9},
{'1': 10, '2': 11, '3': 6, '4': 9, '5': 9, '6': 7, '7': 7, '8': 7, '9': 9, '10': 9, '11': 8, '12': 9, '13': 12, '14': 11, '15': 12, '16': 11},
{'1': 11, '2': 8, '3': 10, '4': 7, '5': 10, '6': 11, '7': 7, '8': 9, '9': 7, '10': 10, '11': 8, '12': 9, '13': 12, '14': 10, '15': 11, '16': 11},
{'1': 10, '2': 10, '3': 9, '4': 10, '5': 9, '6': 9, '7': 9, '8': 8, '9': 6, '10': 10, '11': 8, '12': 10, '13': 12, '14': 10, '15': 12, '16': 10},
{'1': 8, '2': 12, '3': 9, '4': 12, '5': 8, '6': 11, '7': 7, '8': 11, '9': 10, '10': 10, '11': 11, '12': 10, '13': 11, '14': 9, '15': 9, '16': 10}]

human_performances_bad = [{'1': 3, '2': 3, '3': 4, '4': 4, '5': 7, '6': 7, '7': 6, '8': 6, '9': 3, '10': 4, '11': 3, '12': 6, '13': 6, '14': 1, '15': 5, '16': 3},
{'1': 5, '2': 5, '3': 6, '4': 3, '5': 4, '6': 4, '7': 7, '8': 7, '9': 3, '10': 1, '11': 2, '12': 4, '13': 10, '14': 5, '15': 6, '16': 8},
{'1': 2, '2': 2, '3': 3, '4': 5, '5': 6, '6': 4, '7': 6, '8': 8, '9': 5, '10': 4, '11': 4, '12': 6, '13': 10, '14': 6, '15': 8, '16': 8},
{'1': 5, '2': 4, '3': 4, '4': 3, '5': 7, '6': 11, '7': 6, '8': 7, '9': 1, '10': 5, '11': 4, '12': 5, '13': 9, '14': 4, '15': 7, '16': 7},
{'1': 3, '2': 2, '3': 4, '4': 2, '5': 4, '6': 6, '7': 6, '8': 4, '9': 5, '10': 2, '11': 5, '12': 9, '13': 12, '14': 9, '15': 10, '16': 10}]

# ====================== helper functions ======================================

def assign_group_id(row, ps, q_ids):
    '''determine the ID of the problem set a participant (row) completed in the ps-th round'''
    id_ = 1
    # loop over topics i and problem sets j within topics, in order of ps ID
    for i in range(4):
        for j in range(4):
            # check if the first question ID is in topic i set j
            if row['Answer.'+str(ps+1)+'-1-Q'] in q_ids[i][j]:
                return id_
            id_ +=1
    assert False

def compute_scores(row, ps, questions):
    '''count the number of problems that a participant (row) answered correctly in a given ps'''
    num_correct = 0 # start at 1; we model num correct+1 for numerical reasons
    for i in range(12):
        ans = row['Answer.'+str(ps+1)+'-'+str(i+1)+'-A']
        # check participant answer vs. correct answer from questions data
        correct_ans = questions[questions['QID']==row['Answer.'+str(ps+1)+'-'+str(i+1)+'-Q']]['CORRECT'].iloc[0]
        if ans==correct_ans:
            num_correct+=1
    return num_correct

def get_performances(row):
    '''for a given participant (row), return the corresponding other performance,
       i.e., the true scores of the agent the participant evaluated'''
    if row['Input.human_other']==False:
        if row['Input.good']==False:
            return AI_performances_bad
        else:
            return AI_performances_good
    ind = row['Answer.humanPerfIndex']
    if row['Input.good']==False:
        return human_performances_bad[ind]
    else:
        return human_performances_good[ind]

# ====================== data processing ======================================


def preliminary_processing(df, questions):
     # drop rows with missing data and an outlier (index 27)
    to_drop = []
    for j in range(0, len(df)):
        if 'undefined' in [df.iloc[j]['Answer.'+str(i)+'-OtherAssessment'] for i in range(1,17)]:
            to_drop.append(j)
        elif 'undefined' in [df.iloc[j]['Answer.'+str(i)+'-Conf'] for i in range(1,17)]:
            to_drop.append(j)
    df.drop(to_drop, axis=0, inplace=True)
    df = df.drop(27)
    df = df.reset_index(drop=True)

    for i in range(16):
        # for each problem set index i, add a row Y_i to the dataframe corresponding to the number correct
        df['Y_'+str(i+1)] = df.apply(compute_scores, args=(i, questions,), axis=1)

    df.to_csv('processed_data/results.csv') # save intermediate, detailed file in original format
    return df


def get_p_order(df):
    '''generate problem set IDs and create a matrix p_order expressing, for each participant,
       how their problem sets must be reordered to be sorted by problem set ID'''

    # generate a list of, for each topic, a list of sets of question IDs corresponding to each of its four problem sets
    tmp = df.iloc[0]
    q_ids = [[],[],[],[]]
    for i in range(16):
        # get the question IDs in set i for the first participant (tmp)
        question_ids = set(list(tmp[['Answer.'+str(i+1)+"-" + str(j+1)+"-Q" for j in range(12)]]))
        topic = sid_to_topic[str(tmp['Answer.'+str(i+1)+'-SID'])]
        q_ids[topic-1].append(question_ids)

    # problems has columns PID_1 to PID_16 where the value of PID_i in row j corresponds
    # to the problem set ID of the ith problem set participant j completed
    problems = pd.DataFrame()
    for i in range(16):
        # for each problem set index, create a column 'PID_i' representing the PS ID
        problems['PID_'+str(i+1)] = df.apply(assign_group_id, args=(i, q_ids, ), axis=1)

    # generate p_order
    p_order = []
    for row in problems.iterrows():
        order = {}
        counter = 0
        for val in row[1]:
            order[counter] = val
            counter += 1
        p_order.append(order)

    return p_order, problems

def create_modeling_files(df, p_order):
    '''for each of true, self-assessed, and other-assessed performance,
       reorder the scores by p_order and save to a bare-bones csv'''
    
    true_rows = []; self_rows=[]; other_rows=[]
    all_data = [true_rows, self_rows, other_rows]
    cols = range(16)
    names = [("Y_",""),("Answer.","-Conf"),("Answer.","-OtherAssessment")]
    fnames = ["true_data","self_data","other_data"]
    
    # for each participant...
    for row, ind in zip(df.iterrows(), range(len(df))):
        sorted_inds = sorted(cols, key=lambda x:p_order[ind][x])
        for name,r in zip(names,all_data): # for true, self, and other
            # sort data from corresponding columns
            sorted_cols = [name[0]+str(i+1)+name[1] for i in sorted_inds]
            new_row = list(row[1][sorted_cols])
            # add to corresponding list
            r.append([int(x) for x in new_row])

    # convert each list into a dataframe and save as a csv
    for fname, rows in zip(fnames,all_data):
        dat = pd.DataFrame(rows)
        dat = dat.apply(lambda x: x+1) # +1 to all scores for numerical reasons
        dat['feedback'] = df['Answer.feedbackCond']
        dat['human'] = df['Answer.humanCond']
        dat['highacc'] = df['Answer.goodCond']
        dat.to_csv('processed_data/'+fname+'.csv')


def generate_alternate_format(df, problems):
    '''create a csv in a useful data format for exploration/visualization'''

    # expand dataframe to have a column for each participant/problem set combination
    rows = []
    for i in range(len(df)): # over participants
        for j in range(1,17): # over problem sets
            row = []
            to_add = ['WorkerId','Input.human_other','Input.feedback','Input.good','Answer.'+str(j)+'-SID',
                    'Answer.'+str(j)+'-Conf', 'Answer.'+str(j)+'-OtherAssessment','Y_'+str(j)]
            for a in to_add:
                row.append(df.iloc[i][a])
            rows.append(row)

    cols = ['WorkerId', 'HumanCondition','FeedbackCondition','GoodCondition','Topic',
            'SelfAssessment','OtherAssessment','TruePerformance']
    new_df = pd.DataFrame(rows, columns=cols)

    new_df['Topic'] = new_df["Topic"].apply(lambda x: sid_to_topic[str(x)])
    new_df['OtherAssessment'] = new_df['OtherAssessment'].astype(int)
    new_df['SelfAssessment'] = new_df['SelfAssessment'].astype(int)

    # generate a column of corresponding other performance
    result = []
    for i in range(len(df)):
        row = df.iloc[i]
        per = get_performances(row) # get matching other performance
        order = list(problems.iloc[i]) 
        for o in order:
            result.append(per[str(o)]) # reorder scores to match participant's order
    new_df['OtherPerformance'] = result

    # add column of the round of a given problem set (they are already ordered)
    r = [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]*204
    new_df['Round']=r

    new_df.to_csv('processed_data/clean_results.csv')


def order_by_round(df, problems):
    '''generate csvs for modeling, preserving the order participants
       completed the problem sets in & saving order information'''
    true_perf = df[["Y_"+str(i) for i in range(1, 17)]]
    self_assmt = df[["Answer."+ str(i)+"-Conf" for i in range(1,17)]]
    other_assmt = df[["Answer."+ str(i)+"-OtherAssessment" for i in range(1,17)]]

    # create a key for topics from question IDs
    qid_to_topic={}
    qid_list = [[i for i in range(1+j,5+j)] for j in range(0,16,4)]
    for i,j in zip(qid_list,[1,2,3,4]):
        for k in i:
            qid_to_topic[k] = j

    # generate a dataframe corresponding to the topic of each participant/
    # problem set combination, in the order they were completed
    topics = pd.DataFrame()
    for col in problems.columns:
        topics[col] = problems[col].apply(lambda x: qid_to_topic[int(x)])
    topics.to_csv('processed_data/topics.csv')

    for (dat, name) in zip([true_perf, self_assmt, other_assmt],
                            ['true_data_o','self_data_o','other_data_o']):
        dat = dat.astype(int)
        dat = dat.apply(lambda x: x+1)
        dat.to_csv('processed_data/'+name+'.csv')

    problems.to_csv('processed_data/problems.csv')


def main():

    # read experimental results files
    data_files = ['raw_data/main_batches/batch'+str(i)+'.csv' for i in range (1,5)]
    dfs = []
    for f in data_files:
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs)
    df.reset_index(drop=True, inplace=True)

    # read trivia question data
    questions = pd.read_csv('raw_data/trivia_questions.csv', index_col=0)

    # preliminary data processing and problem set ordering info
    df = preliminary_processing(df, questions)
    p_order, problems = get_p_order(df)

    # process data and save csvs
    create_modeling_files(df, p_order)
    generate_alternate_format(df.copy(), problems)
    order_by_round(df.copy(), problems)

if __name__=="__main__":
    main()