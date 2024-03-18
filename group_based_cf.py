import pandas as pd
import numpy as np
import os
from user_based_cf import getRecommendedItems
from user_based_cf import basePred

def addUsersPred(df, users, items):
    for user in users:
        for item in items:
            uRating = df[(df['userId'] == user) & (df['movieId'] == item)]['rating']
            
            if uRating.empty:
                newRow = {'userId': user, 'movieId': item, 'rating': basePred(df, user, item),  'timestamp': 0}
                df.loc[len(df)] = newRow

def getGroupAveragePred(df, items, users, k=10):
    item2pred = []
    for item in items:
        sum = 0
        for user in users:
            uRating = df[(df['userId'] == user) & (df['movieId'] == item)]['rating']

            if uRating.empty:
                uRating = basePred(df, user, item)
            else:
                uRating = uRating.values[0]
            
            sum += uRating
            
        item2pred.append((item, sum / len(users)))

    return sorted(item2pred, key=lambda x: x[1], reverse=True)[:k]

def getGroupLeastMiseryPred(df, items, users, k=10):
    item2pred = []
    for item in items:
        min = np.inf
        for user in users:
            uRating = df[(df['userId'] == user) & (df['movieId'] == item)]['rating']

            if uRating.empty:
                uRating = basePred(df, user, item)
            else:
                uRating = uRating.values[0]
                
            if uRating < min:
                min = uRating
            
        item2pred.append((item, min))

    return sorted(item2pred, key=lambda x: x[1], reverse=True)[:k]

def getUserPred(df, user, items, k=10):
    userPred = []

    for item in items:
        uRating = df[(df['userId'] == user) & (df['movieId'] == item)]['rating']

        if uRating.empty:
            uRating = basePred(df, user, item)
        else:
            uRating = uRating.values[0]
    
        userPred.append(uRating)
    
    return sorted(userPred, key=lambda x: x, reverse=True)[:k]

def getSatisfaction(df, items, user):
    den = sum(getUserPred(df, user, items, len(items)))
    
    for item in items:
        num = 0
            
        uRating = df[(df['userId'] == user) & (df['movieId'] == item)]['rating']

        if uRating.empty:
            uRating = basePred(df, user, item)
        else:
            uRating = uRating.values[0]
            
        num += uRating
    
    return num / den

def getSequentialRecommendations(df, items, users, k=10):
    items = [x[0] for x in getGroupAveragePred(df, items, users, int(len(items)/2))]
    candidateSet = [items.pop(0)]

    for _ in range(0, k):
        min = np.inf
        bestItem = None

        for item in items:
            satisfaction = 0
            tmp = candidateSet.copy()
            tmp.append(item)
            
            for i in range(0, len(users)):
                for j in range(i+1, len(users)):
                    satisfaction += abs(getSatisfaction(df, tmp, users[i]) - getSatisfaction(df, tmp, users[j]))

            if satisfaction < min:
                min = satisfaction
                bestItem = item

        items.remove(bestItem)
        candidateSet.append(bestItem)
    
    return candidateSet

def main():
    dfPath = os.path.join(os.getcwd(), 'group_recommendations', 'ml-latest-small', 'ratings.csv')
    df = pd.read_csv(dfPath)
    print(df.head())
    print("\nRow num: ", df.shape[0])

    users = [1, 2, 5]

    items = []
    for user in users:
        uPred = getRecommendedItems(df, user)
        items.extend([x[0] for x in uPred])
    print(len(items))

    addUsersPred(df, users, items)

    print(getSequentialRecommendations(df, items, users, k=10))

if __name__ == "__main__":
    main()