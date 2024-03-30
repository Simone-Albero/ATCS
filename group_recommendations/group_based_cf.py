import progressbar as pb
import pandas as pd
import numpy as np
import os

from user_based_cf import getRecommendedItems
from user_based_cf import recursivePred
from user_based_cf import pearsonSimilarity

def getSimUser(df, user, k = 1, sim_th = 0.7):
    candidates = []
    for candidate in df['userId'].unique():
        sim = pearsonSimilarity(df, candidate, user)
        if sim < sim_th or candidate == user: continue

        if k == 1: return candidate

        candidates.append(candidate)
        if len(candidates) == k: return candidates
    
    return None

def getDissUser(df, user, k = 1, diss_th = -0.7):
    candidates = []
    for candidate in df['userId'].unique():
        sim = pearsonSimilarity(df, candidate, user)
        if sim > diss_th or candidate == user: continue

        if k == 1: return candidate

        candidates.append(candidate)
        if len(candidates) == k: return candidates
    
    return None

def itemToUserRating(df, item, user):
    rating = df[(df['userId'] == user) & (df['movieId'] == item)]['rating']

    if rating.empty:
        rating = recursivePred(df, user, item)
    else:
        rating = rating.values[0]
    
    return rating

def itemsToUserScore(df, items, user):
    items_to_score = {}

    for item in items:
        items_to_score[item] = itemToUserRating(df, item, user)
    
    items_to_score = sorted(items_to_score.items(), key=lambda x: x[1])
    items_to_score = {key_value[0]: posizione for posizione, key_value in enumerate(items_to_score, 1)}
    
    return  items_to_score

def topRatings(df, items, user, k=10):
    ratings = []
    
    for item in items:
        ratings.append(itemToUserRating(df, item, user))
    
    return sorted(ratings, reverse=True)[:k]

def generateUsersRatings(df, items, users):
    headers = ['userId', 'movieId', 'rating']
    ratings_u = pd.DataFrame(columns=headers)
    
    with pb.ProgressBar(max_value=len(users)*len(items)) as bar:
        for user in users:
            for item in items:
                rating = itemToUserRating(df, item, user)
                new_row = [user, item, rating]
                ratings_u.loc[len(ratings_u)] = new_row
                bar.next()
    
    return ratings_u

def getAverageScore(df, item, users):
    sum = 0
    for user in users:
        u_rating = itemToUserRating(df, item, user)         
        sum += u_rating
            
    return sum / len(users)

def groupAveragePred(df, items, users, k=10):
    item_to_pred = []

    for item in items:
        item_to_pred.append(getAverageScore(df, item, users))

    return sorted(item_to_pred, key=lambda x: x[1], reverse=True)[:k]

def getLeastScore(df, item, users):
    min_score = np.inf
    for user in users:
        u_rating = itemToUserRating(df, item, user)
                
        if u_rating < min_score:
            min_score = u_rating
    
    return min_score

def groupLeastMiseryPred(df, items, users, k=10):
    item_to_pred = []

    for item in items:        
        item_to_pred.append(getLeastScore(df, item, users))

    return sorted(item_to_pred, key=lambda x: x[1], reverse=True)[:k]

def getSatisfaction(df, items, group_items, user):
    den = np.sum(topRatings(df, items, user, len(group_items)))
    
    num = 0
    for item in group_items:
        u_rating = itemToUserRating(df, item, user)
        num += u_rating
    
    return num / den

def sequentialRecommendations(df, items, users, k=10):
    items = [x[0] for x in groupAveragePred(df, items, users, len(items))]
    candidate_set = [items.pop(0)]

    for _ in range(k-1):
        min = np.inf
        best_item = None

        for item in items:
            satisfaction = 0
            tmp_set = candidate_set.copy()
            tmp_set.append(item)
            
            for i in range(0, len(users)):
                for j in range(i+1, len(users)):
                    satisfaction += abs(getSatisfaction(df, items, tmp_set, users[i]) - getSatisfaction(df, items, tmp_set, users[j]))

            if satisfaction < min:
                min = satisfaction
                best_item = item

        items.remove(best_item)
        candidate_set.append(best_item)
    
    return candidate_set

def customRecommendations(df, items, users, k=10):
    users_scores = {}
    for user in users:
        users_scores[user] = itemsToUserScore(df, items, user)

    items_score = []
    for item in items:
        disagreement, global_rank = 1, 0

        for user in users:
            global_rank += users_scores[user][item]

        for i in range(0, len(users)):
            for j in range(i+1, len(users)):
                disagreement += abs(users_scores[users[i]][item] - users_scores[users[j]][item])

        score = disagreement / global_rank
        items_score.append((item, score))

    return sorted(items_score, key=lambda x: x[1])[:k]

def main():
    df_path = os.path.join(os.getcwd(), 'group_recommendations', 'dataset', 'ratings.csv')
    df = pd.read_csv(df_path)
    
    users = [17]
    users.append(getSimUser(df, users[0]))
    users.append(getDissUser(df, users[0]))

    print(users)

    items = set()
    for user in users:
        u_items = getRecommendedItems(df, user, 3, 4, 20, 4)
        print(u_items)
        items.update([x[0] for x in u_items])

    ratings = generateUsersRatings(df, items, users)
    print(ratings)
    
    headers = ['pred_fun',  'user1_sat', 'user2_sat', 'user3_sat']
    stats = pd.DataFrame(columns=headers)

    recommended_items = [x[0] for x in groupAveragePred(ratings, items, users, 4)]
    new_row = ['groupAveragePred', getSatisfaction(ratings, items, recommended_items, users[0]), getSatisfaction(ratings, items, recommended_items, users[1]), getSatisfaction(ratings, items, recommended_items, users[2])]
    stats.loc[len(stats)] = new_row

    recommended_items = [x[0] for x in groupLeastMiseryPred(ratings, items, users, 4)]
    new_row = ['groupLeastMiseryPred', getSatisfaction(ratings, items, recommended_items, users[0]), getSatisfaction(ratings, items, recommended_items, users[1]), getSatisfaction(ratings, items, recommended_items, users[2])]
    stats.loc[len(stats)] = new_row

    recommended_items = [x[0] for x in customRecommendations(ratings, items, users, 4)]
    new_row = ['customRecommendations', getSatisfaction(ratings, items, recommended_items, users[0]), getSatisfaction(ratings, items, recommended_items, users[1]), getSatisfaction(ratings, items, recommended_items, users[2])]
    stats.loc[len(stats)] = new_row

    recommended_items = sequentialRecommendations(ratings, items, users, 4)
    new_row = ['sequentialRecommendations', getSatisfaction(ratings, items, recommended_items, users[0]), getSatisfaction(ratings, items, recommended_items, users[1]), getSatisfaction(ratings, items, recommended_items, users[2])]
    stats.loc[len(stats)] = new_row

    print(stats)
    #stats.to_csv('group_sat.csv', index=False)

def test():
    df_path = os.path.join(os.getcwd(), 'group_recommendations', 'dataset', 'test.csv')
    df = pd.read_csv(df_path)

    users = [1, 2, 3]
    items = list(range(1, 13))

    headers = ['pred_fun',  'user1_sat', 'user2_sat', 'user3_sat']
    stats = pd.DataFrame(columns=headers)

    recommended_items = [x[0] for x in groupAveragePred(df, items, users, 4)]
    print(recommended_items)
    new_row = ['groupAveragePred', getSatisfaction(df, items, recommended_items, users[0]), getSatisfaction(df, items, recommended_items, users[1]), getSatisfaction(df, items, recommended_items, users[2])]
    stats.loc[len(stats)] = new_row

    recommended_items = [x[0] for x in groupLeastMiseryPred(df, items, users, 4)]
    print(recommended_items)
    new_row = ['groupLeastMiseryPred', getSatisfaction(df, items, recommended_items, users[0]), getSatisfaction(df, items, recommended_items, users[1]), getSatisfaction(df, items, recommended_items, users[2])]
    stats.loc[len(stats)] = new_row
    
    recommended_items = [x[0] for x in customRecommendations(df, items, users, 4)]
    print(recommended_items)
    new_row = ['customRecommendations', getSatisfaction(df, items, recommended_items, users[0]), getSatisfaction(df, items, recommended_items, users[1]), getSatisfaction(df, items, recommended_items, users[2])]
    stats.loc[len(stats)] = new_row

    recommended_items = sequentialRecommendations(df, items, users, 4)
    print(recommended_items)
    new_row = ['sequentialRecommendations', getSatisfaction(df, items, recommended_items, users[0]), getSatisfaction(df, items, recommended_items, users[1]), getSatisfaction(df, items, recommended_items, users[2])]
    stats.loc[len(stats)] = new_row

    print(stats)
    stats.to_csv('group_sat.csv', index=False)

if __name__ == "__main__":
    main()