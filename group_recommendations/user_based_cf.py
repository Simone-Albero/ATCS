import progressbar as pb
import numpy as np
import pandas as pd
import random
import time
import os
import timeit

def getCoRatedItems(df, user_x, user_y):
    ratings_x = df[df['userId'] == user_x]
    ratings_y = df[df['userId'] == user_y]

    return pd.merge(ratings_x, ratings_y, on='movieId', how='inner')

def pearsonSimilarity(df, user_x, user_y):
    corated_items = getCoRatedItems(df, user_x, user_y)
    if corated_items.empty: return 0

    co_ratings_x, co_ratings_y = corated_items['rating_x'], corated_items['rating_y']
    ratings_x, ratings_y = df[df['userId'] == user_x]['rating'], df[df['userId'] == user_y]['rating']
    mean_x, mean_y = np.mean(ratings_x), np.mean(ratings_y)

    den = np.sqrt(np.sum(np.square(co_ratings_x - mean_x))) * np.sqrt(np.sum(np.square(co_ratings_y - mean_y)))
    
    if den == 0: return 0

    PENALITY_TH = 5
    penality_factor = 1 if corated_items.shape[0] >= PENALITY_TH else corated_items.shape[0] / PENALITY_TH
    return penality_factor * np.sum((co_ratings_x - mean_x) * (co_ratings_y - mean_y)) / den

def cosineSimilarity(df, user_x, user_y):
    corated_items = getCoRatedItems(df, user_x, user_y)
    if corated_items.empty: return 0

    co_ratings_x, co_ratings_y = corated_items['rating_x'], corated_items['rating_y']
 
    num = np.dot(co_ratings_x, co_ratings_y)
    den = np.linalg.norm(co_ratings_x) * np.linalg.norm(co_ratings_y)

    if den == 0: return 0
    return num / den

def jaccardSimilarity(df, user_x, user_y):
    corated_items = getCoRatedItems(df, user_x, user_y)
    if corated_items.empty: return 0

    co_ratings_x, co_ratings_y = set(corated_items['rating_x']), set(corated_items['rating_y'])

    intersection = len(co_ratings_x.intersection(co_ratings_y))
    union = len(co_ratings_x.union(co_ratings_y))

    if union == 0: return 0
    return intersection / union

def euclideanDistance(df, user_x, user_y):
    corated_items = getCoRatedItems(df, user_x, user_y)
    if corated_items.empty: return 0

    co_ratings_x, co_ratings_y = corated_items['rating_x'], corated_items['rating_y']

    return 1 / (1 + np.sqrt(np.sum(np.square(co_ratings_x - co_ratings_y))))

def manhattanDistance(df, user_x, user_y):
    corated_items = getCoRatedItems(df, user_x, user_y)
    if corated_items.empty: return 0

    co_ratings_x, co_ratings_y = corated_items['rating_x'], corated_items['rating_y']

    return 1 / (1 + np.sum(np.abs(co_ratings_x - co_ratings_y)))


def getNeighbors(df, user, item = None, blacklist = [], sim_fun = pearsonSimilarity, k = 20):
    candidates = df['userId'].unique()
    candidates = pd.DataFrame({'userId': candidates})
    candidates = candidates[(candidates['userId'] != user) & (~candidates['userId'].isin(blacklist))]
    if item != None: candidates = candidates[candidates['userId'].apply(lambda candidate: ((df['userId'] == candidate) & (df['movieId'] == item)).any())]

    candidates['sim'] = candidates['userId'].apply(lambda candidate: sim_fun(df, user, candidate))
    candidates = candidates.sort_values(by='sim', key=lambda x: abs(x), ascending=False)

    return candidates.to_records(index=False).tolist()[:k]

def customGetNeighbors(df, user, item, blacklist = [], sim_fun = pearsonSimilarity, k1 = 10, k2 = 10):
    neighbors = []

    item_based_neighbors = getNeighbors(df, user, item, blacklist, sim_fun, k1)
    neighbors.extend(item_based_neighbors)

    sim_based_neighbors = getNeighbors(df, user, None, blacklist, sim_fun, k2)
    neighbors.extend(sim_based_neighbors)
    
    return neighbors

def neighborFactor(df, neighbor, sim, item):
    rating = df[(df['userId'] == neighbor) & (df['movieId'] == item)]['rating'].values[0]
    mean = df[df['userId'] == neighbor]['rating'].mean()

    return sim * (rating - mean)

def basePred(df, user, item, sim_fun = pearsonSimilarity, k = 20):
    neighbors = getNeighbors(df, user, item, [], sim_fun, k)
    mean_u = np.mean(df[df['userId'] == user]['rating'])
    num, den = 0, 0

    neighbors = pd.DataFrame(neighbors, columns=['neighbor', 'sim'])
    if neighbors.empty: return 0
    neighbors['factor_n'] = neighbors.apply(lambda row: neighborFactor(df, row.iloc[0], row.iloc[1], item), axis=1)

    num = np.sum(neighbors['factor_n'])
    den = np.sum(np.abs(neighbors['sim']))

    if den == 0: return 0
    return mean_u + num / den


def recursivePred(df, user, item, lev = 0, blacklist = [], sim_fun = pearsonSimilarity, k1 = 10, k2 = 10, lmb = 0.8, lev_th = 1):
    if lev >= lev_th:
        return basePred(df, user, item, sim_fun, k1)

    neighbors = customGetNeighbors(df, user, item, blacklist, sim_fun, k1, k2)
    mean_u = np.mean(df[df['userId'] == user]['rating'])
    num, den = 0, 0
    
    for neighbor, sim in neighbors:
        ratings_n = df[(df['userId'] == neighbor) & (df['movieId'] == item)]['rating']
        mean_n = np.mean(df[df['userId'] == neighbor]['rating'])

        if not ratings_n.empty:
            num += sim * (ratings_n.values[0] - mean_n)
            den += abs(sim)
        else:
            blacklist.append(user)
            num += lmb * sim * (recursivePred(df, neighbor, item, lev+1, blacklist.copy(), sim_fun, k1, k2, lmb, lev_th) - mean_n)
            den += lmb * abs(sim)            

    if den == 0 or neighbors == []: return 0
    return mean_u + num / den
    
def getRecommendedItems(df, user, rate_th = 4.5, pred_th = 4, max_neighbors = 10, k = 10, sim_fun = pearsonSimilarity):
    u_items = set(df[df['userId'] == user]['movieId'])
    
    neighbors = getNeighbors(df, user, None, [], sim_fun, max_neighbors)
    
    n_items = set()
    for neighbor, _ in neighbors:
        items = set(df[(df['userId'] == neighbor) & (df['rating'] >= rate_th)]['movieId'])
        n_items.update(items)

    not_yet_rated = list(n_items - u_items)
    random.shuffle(not_yet_rated)

    item_to_pred = []

    with pb.ProgressBar(max_value=k) as bar:
        for item in not_yet_rated:
            pred = recursivePred(df, user, item)

            if pred >= pred_th:
                item_to_pred.append((item, pred))
                bar.next()
            if pred_th != 0 and len(item_to_pred) == k:
                break

    return sorted(item_to_pred, key=lambda x: x[1], reverse=True)[:k]

def getRandomSample(df, itemsNum = 10, usersNum = 10, seed = 42):
    np.random.seed(seed)
    samples = []
    users = np.random.choice(df['userId'].unique(), usersNum, replace=False)

    for user in users:
        uItems = list(df[df['userId'] == user]['movieId'])
        items = np.random.choice(uItems, itemsNum, replace=False)

        sample = [(user, item) for item in items]
        samples += sample
    
    return samples

def evaluatePred(df, sample, pred_fun = recursivePred, k1 = 20, k2 = 10, sim_fun = pearsonSimilarity, lmb = 0.5, lev_th = 1):
    err = []
    tot_time = 0

    with pb.ProgressBar(max_value=len(sample)) as bar:
        for user, item in sample:
            real_pred = df[(df['userId'] == user) & (df['movieId'] == item)]['rating'].values[0]

            if pred_fun.__name__ == 'basePred':
                start_time = time.time()
                generated_pred = pred_fun(df, user, item, sim_fun, k1)
                end_time = time.time()
                tot_time += end_time - start_time
            else:
                start_time = time.time()
                generated_pred = pred_fun(df, user, item, 0, [], sim_fun, k1, k2, lmb, lev_th)
                end_time = time.time()
                tot_time += end_time - start_time

            err.append(abs(generated_pred - real_pred))
            bar.next()

    return np.mean(err), np.std(err), np.max(err), tot_time/len(sample)

def test():
    df_path = os.path.join(os.getcwd(), 'group_recommendations', 'dataset', 'ratings.csv')
    df = pd.read_csv(df_path)

    sample = getRandomSample(df, 5, 10)

    headers = ['pred_fun', 'sim_fun', 'k1', 'k2', 'lmb', 'lev_th', 'mean_err', 'std_err', 'max_err', 'mean_time']

    stats = pd.DataFrame(columns=headers) # Evaluating k1 on basePred
    for k1 in np.arange(5, 35, 5):
        mean_err, std_err, max_err, mean_time = evaluatePred(df, sample, basePred, k1)
        new_row = ['basePred', 'pearsonSimilarity', k1, np.nan, np.nan, np.nan, mean_err, std_err, max_err, mean_time]
        stats.loc[len(stats)] = new_row

    print(stats)
    stats.to_csv('k1.csv', index=False)
    
    # stats = pd.DataFrame(columns=headers) # Evaluating similarities on basePred
    # K1 = 20
    # for sim_fun in [pearsonSimilarity, cosineSimilarity, jaccardSimilarity, euclideanDistance, manhattanDistance]:
    #     mean_err, std_err, max_err, mean_time = evaluatePred(df, sample, basePred, K1, None, sim_fun)
    #     new_row = ['basePred', sim_fun.__name__, K1, np.nan, np.nan, np.nan, mean_err, std_err, max_err, mean_time]
    #     stats.loc[len(stats)] = new_row

    # print(stats)
    # stats.to_csv('similarities.csv', index=False)

    
    # stats = pd.DataFrame(columns=headers) # Evaluating k2 on recursivePred
    # K1, LMB, LEV_TH = 15, 0.3, 1
    # for k2 in np.arange(5, 25, 5):
    #     mean_err, std_err, max_err, mean_time = evaluatePred(df, sample, recursivePred, K1, k2, pearsonSimilarity, LMB, LEV_TH)
    #     new_row = ['recursivePred', 'pearsonSimilarity', K1, k2, LMB, LEV_TH, mean_err, std_err, max_err, mean_time]
    #     stats.loc[len(stats)] = new_row

    # print(stats)
    # stats.to_csv('k2.csv', index=False)

    
    # stats = pd.DataFrame(columns=headers) # Evaluating lmb on recursivePred
    # K1, K2, LEV_TH = 10, 10, 1
    # for lmb in np.arange(0.1, 1.1, 0.1):
    #     mean_err, std_err, max_err, mean_time = evaluatePred(df, sample, recursivePred, K1, K2, pearsonSimilarity, lmb, LEV_TH)
    #     new_row = ['recursivePred', 'pearsonSimilarity', K1, K2, lmb, LEV_TH, mean_err, std_err, max_err, mean_time]
    #     stats.loc[len(stats)] = new_row

    # print(stats)
    # stats.to_csv('lmb.csv', index=False)

    
    # stats = pd.DataFrame(columns=headers) # Evaluating lev_th on recursivePred
    # K1, K2, LMB= 20, 10, 0.3
    # for lev_th in np.arange(0, 3, 1):
    #     mean_err, std_err, max_err, mean_time = evaluatePred(df, sample, recursivePred, K1, K2, pearsonSimilarity, LMB, lev_th)
    #     new_row = ['recursivePred', 'pearsonSimilarity', K1, K2, LMB, lev_th, mean_err, std_err, max_err, mean_time]
    #     stats.loc[len(stats)] = new_row

    # print(stats)
    # stats.to_csv('lev_th.csv', index=False)

def main():
    df_path = os.path.join(os.getcwd(), 'group_recommendations', 'dataset', 'ratings.csv')
    df = pd.read_csv(df_path)

    print(getNeighbors(df, 1))
    #print(timeit.timeit(lambda: getRecommendedItems(df, 1), number=5))




if __name__ == "__main__":
    test()