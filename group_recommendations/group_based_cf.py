import progressbar as pb
import pandas as pd
import numpy as np
import os
from user_based_cf import getRecommendedItems
from user_based_cf import basePred

def itemToRating(df, items, user, k=10):
    user_pred = []

    for item in items:
        u_ratings = df[(df['userId'] == user) & (df['movieId'] == item)]['rating']

        if u_ratings.empty:
            u_ratings = (item, basePred(df, user, item))
        else:
            u_ratings = (item, u_ratings.values[0])
    
        user_pred.append(u_ratings)
    
    return sorted(user_pred, key=lambda x: x[1], reverse=True)[:k]

def getUserRatings(df, items, user, k=10):
    return [x[1] for x in itemToRating(df, items, user, k)]

def getUserRating(df, item, user):
    return itemToRating(df, [item], user)[0][1]


def generateUsersRatings(df, items, users):
    headers = ['userId', 'movieId', 'rating']
    u_ratings = pd.DataFrame(columns=headers)
    
    with pb.ProgressBar(max_value=len(users)*len(items)) as bar:
        for user in users:
            for item in items:
                u_rating = getUserRating(df, item, user)
                new_row = [user, item, u_rating]
                u_ratings.loc[len(u_ratings)] = new_row
                bar.next()
    
    return u_ratings

def groupAveragePred(df, items, users, k=10):
    item_to_pred = []

    for item in items:
        sum = 0
        for user in users:
            u_rating = getUserRating(df, item, user)         
            sum += u_rating
            
        item_to_pred.append((item, sum / len(users)))

    return sorted(item_to_pred, key=lambda x: x[1], reverse=True)[:k]

def groupLeastMiseryPred(df, items, users, k=10):
    item_to_pred = []

    for item in items:
        min = np.inf
        for user in users:
            u_rating = getUserRating(df, item, user)
                
            if u_rating < min:
                min = u_rating
            
        item_to_pred.append((item, min))

    return sorted(item_to_pred, key=lambda x: x[1], reverse=True)[:k]

def getSatisfaction(df, items, candidate_set, user):
    den = sum(getUserRatings(df, items, user, len(candidate_set)))
    
    num = 0
    for candidate in candidate_set:
        u_rating = getUserRating(df, candidate, user)
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

def itemToScore(df, items, user, k=10):
    items_to_score = {}
    u_ratings = itemToRating(df, items, user, k)

    score = len(u_ratings)
    for rating in u_ratings:
        items_to_score[rating[0]] = score
        score -= 1

    return items_to_score

def customRecommendations(df, items, users, k=10):
    users_scores = {}
    for user in users:
        users_scores[user] = itemToScore(df, items, user, len(items))

    items_rank = []
    for item in items:
        disagreement, global_rank = 1, 0

        for user in users:
            global_rank += users_scores[user][item]

        for i in range(0, len(users)):
            for j in range(i+1, len(users)):
                disagreement += abs(users_scores[users[i]][item] - users_scores[users[j]][item])

        score = disagreement / global_rank
        items_rank.append((item, score))

    return sorted(items_rank, key=lambda x: x[1])[:k]

def main():
    df_path = os.path.join(os.getcwd(), 'group_recommendations', 'dataset', 'ratings.csv')
    df = pd.read_csv(df_path)
    
    np.random.seed(42)
    users = np.random.choice(df['userId'].unique(), 3, replace=False)

    items = set()
    for user in users:
        u_items = getRecommendedItems(df, user, 4.5, 4, 10, 5)
        items.update([x[0] for x in u_items])

    u_ratings = generateUsersRatings(df, items, users)
    
    headers = ['pred_fun',  'user1_sat', 'user2_sat', 'user3_sat']
    stats = pd.DataFrame(columns=headers)

    recommended_items = [x[0] for x in groupAveragePred(u_ratings, items, users, 4)]
    new_row = ['groupAveragePred', getSatisfaction(u_ratings, items, recommended_items, users[0]), getSatisfaction(u_ratings, items, recommended_items, users[1]), getSatisfaction(u_ratings, items, recommended_items, users[2])]
    stats.loc[len(stats)] = new_row

    recommended_items = [x[0] for x in groupLeastMiseryPred(u_ratings, items, users, 4)]
    new_row = ['groupLeastMiseryPred', getSatisfaction(u_ratings, items, recommended_items, users[0]), getSatisfaction(u_ratings, items, recommended_items, users[1]), getSatisfaction(u_ratings, items, recommended_items, users[2])]
    stats.loc[len(stats)] = new_row

    recommended_items = [x[0] for x in customRecommendations(u_ratings, items, users, 4)]
    new_row = ['customRecommendations', getSatisfaction(u_ratings, items, recommended_items, users[0]), getSatisfaction(u_ratings, items, recommended_items, users[1]), getSatisfaction(u_ratings, items, recommended_items, users[2])]
    stats.loc[len(stats)] = new_row

    recommended_items = sequentialRecommendations(u_ratings, items, users, 4)
    new_row = ['sequentialRecommendations', getSatisfaction(u_ratings, items, recommended_items, users[0]), getSatisfaction(u_ratings, items, recommended_items, users[1]), getSatisfaction(u_ratings, items, recommended_items, users[2])]
    stats.loc[len(stats)] = new_row

    print(stats)
    stats.to_csv('group_sat.csv', index=False)

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