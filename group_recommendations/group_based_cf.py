import pandas as pd
import numpy as np
import os
from user_based_cf import getRecommendedItems
from user_based_cf import basePred

def getUserPreds(df, user, items, k=10):
    user_pred = []

    for item in items:
        u_ratings = df[(df['userId'] == user) & (df['movieId'] == item)]['rating']

        if u_ratings.empty:
            u_ratings = basePred(df, user, item)
        else:
            u_ratings = u_ratings.values[0]
    
        user_pred.append(u_ratings)
    
    return sorted(user_pred, key=lambda x: x, reverse=True)[:k]

def generateUsersRatings(df, users, items):
    headers = ['userId', 'movieId', 'rating']
    u_ratings = pd.DataFrame(columns=headers)
    
    for user in users:
        for item in items:
            u_rating = getUserPreds(df, user, [item])[0]
            new_row = [user, item, u_rating]
            u_ratings.loc[len(u_ratings)] = new_row
    
    return u_ratings

def getGroupAveragePred(df, items, users, k=10):
    item_to_pred = []

    for item in items:
        sum = 0
        for user in users:
            u_rating = getUserPreds(df, user, [item])[0]            
            sum += u_rating
            
        item_to_pred.append((item, sum / len(users)))

    return sorted(item_to_pred, key=lambda x: x[1], reverse=True)[:k]

def getGroupLeastMiseryPred(df, items, users, k=10):
    item_to_pred = []

    for item in items:
        min = np.inf
        for user in users:
            u_rating = getUserPreds(df, user, [item])[0]      
            sum += u_rating
                
            if u_rating < min:
                min = u_rating
            
        item_to_pred.append((item, min))

    return sorted(item_to_pred, key=lambda x: x[1], reverse=True)[:k]

def getSatisfaction(df, items, user):
    den = sum(getUserPreds(df, user, items, len(items)))
    
    num = 0
    for item in items:
        u_rating = getUserPreds(df, user, [item])[0]
        num += u_rating
    
    return num / den

def getSequentialRecommendations(df, items, users, k=10):
    items = [x[0] for x in getGroupAveragePred(df, items, users, int(len(items)/2))]
    candidate_set = [items.pop(0)]

    for _ in range(0, k):
        min = np.inf
        best_item = None

        for item in items:
            satisfaction = 0
            tmp = candidate_set.copy()
            tmp.append(item)
            
            for i in range(0, len(users)):
                for j in range(i+1, len(users)):
                    satisfaction += abs(getSatisfaction(df, tmp, users[i]) - getSatisfaction(df, tmp, users[j]))

            if satisfaction < min:
                min = satisfaction
                best_item = item

        items.remove(best_item)
        candidate_set.append(best_item)
    
    return candidate_set

def main():
    df_path = os.path.join(os.getcwd(), 'group_recommendations', 'ml-latest-small', 'ratings.csv')
    df = pd.read_csv(df_path)
    users = [1, 2, 5]

    items = []
    for user in users:
        u_items = getRecommendedItems(df, user)
        items.extend([x[0] for x in u_items])

    u_ratings = generateUsersRatings(df, users, items)

    print(getSequentialRecommendations(u_ratings, items, users, k=10))

if __name__ == "__main__":
    main()