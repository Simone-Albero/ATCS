import progressbar as pb
import pandas as pd
import numpy as np
import os

from group_based_cf import getSatisfaction
from group_based_cf import getAverageScore
from group_based_cf import getLeastScore
from group_based_cf import getSimUser
from group_based_cf import getDissUser
from group_based_cf import generateUsersRatings

from user_based_cf import getRecommendedItems

def overallSatisfaction(df, items, iters_items, user):
    stats = [getSatisfaction(df, items, iter_items, user) for iter_items in iters_items]
    return np.mean(stats)

def groupSatisfaction(df, items, group_items, users):
    stats = [getSatisfaction(df, items, group_items, user) for user in users]
    return np.mean(stats)

def overallGroupSatisfaction(df, items, iters_items, users):
    stats = [overallSatisfaction(df, items, iters_items, user) for user in users]
    return np.mean(stats)
    
def groupDisagreements(df, items, iters_items, users):
    o_sats = [overallSatisfaction(df, items, iters_items, user) for user in users]

    max_sat, min_sat = np.max(o_sats), np.min(o_sats)
    return max_sat - min_sat

def hybridAggregationScore(df, item, users, alpha):
    return (1 - alpha) * getAverageScore(df, item, users) + alpha * getLeastScore(df, item, users)

def sequentialHybridAggregation(df, items, users, num_iters = 3, k = 10):

    iters_items = []
    alpha = 0
    for i in range(num_iters):
        iter_scores = [(item, hybridAggregationScore(df, item, users, alpha)) for item in items]
        iters_items.append([x[0] for x in sorted(iter_scores, key=lambda x: x[1], reverse=True)[:k]])
        print(iters_items)
        print(groupSatisfaction(df, items, iters_items[i], users))
        print(groupDisagreements(df, items, iters_items, users))

        alpha = groupDisagreements(df, items, iters_items, users)

def main():
    df_path = os.path.join(os.getcwd(), 'group_recommendations', 'dataset', 'ratings.csv')
    df = pd.read_csv(df_path)
    
    #ITER_NUM = 3
    #df_chunks = np.array_split(df, ITER_NUM)

    users = [17]
    users.append(getSimUser(df, users[0]))
    users.append(getDissUser(df, users[0]))

    items = set()
    for user in users:
        items_u = getRecommendedItems(df, user, 3, 4, 20, 4)
        print(items_u)
        items.update([x[0] for x in items_u])

    ratings = generateUsersRatings(df, items, users)

    sequentialHybridAggregation(ratings, items, users)

def test():
    df_path = os.path.join(os.getcwd(), 'group_recommendations', 'dataset', 'test.csv')
    df = pd.read_csv(df_path)

    users = [1, 2, 3]
    items = list(range(1, 13))

    sequentialHybridAggregation(df, items, users)


if __name__ == "__main__":
    test()