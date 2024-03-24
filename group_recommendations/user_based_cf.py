import progressbar as pb
import numpy as np
import pandas as pd
import random
import time
import os

def getCoRatedItems(df, user_x, user_y):
    x_ratings = df[df['userId'] == user_x]
    y_ratings = df[df['userId'] == user_y]

    return pd.merge(x_ratings, y_ratings, on='movieId', how='inner')

def pearsonSimilarity(df, user_x, user_y):
    corated_items = getCoRatedItems(df, user_x, user_y)

    if corated_items.empty:
        return 0

    x_ratings = corated_items['rating_x']
    y_ratings = corated_items['rating_y']

    x_mean = np.mean(x_ratings)
    y_mean = np.mean(y_ratings)

    den = np.sqrt(np.sum((x_ratings - x_mean)**2)) * np.sqrt(np.sum((y_ratings - y_mean)**2))
    if den == 0:
        return 0
    else:
        return np.sum((x_ratings - x_mean) * (y_ratings - y_mean)) / den
    
def cosineSimilarity(df, user_x, user_y):
    corated_items = getCoRatedItems(df, user_x, user_y)

    if corated_items.empty:
        return 0

    x_ratings = corated_items['rating_x']
    y_ratings = corated_items['rating_y']

    mean = np.mean(df[df['movieId'].isin(corated_items['movieId'])]['rating'])
    
    num = np.dot(x_ratings - mean, y_ratings - mean)
    den = np.linalg.norm(x_ratings - mean) * np.linalg.norm(y_ratings - mean)

    if den == 0:
        return 0
    else:
        return num / den

def jaccardSimilarity(df, user_x, user_y):
    corated_items = getCoRatedItems(df, user_x, user_y)

    if corated_items.empty:
        return 0

    x_ratings = set(corated_items['rating_x'])
    y_ratings = set(corated_items['rating_y'])

    intersection = len(x_ratings.intersection(y_ratings))
    union = len(x_ratings.union(y_ratings))

    if union == 0:
        return 0
    return intersection / union

def euclideanDistance(df, user_x, user_y):
    corated_items = getCoRatedItems(df, user_x, user_y)

    if corated_items.empty:
        return 0

    x_ratings = corated_items['rating_x']
    y_ratings = corated_items['rating_y']

    return 1 / (1 + np.sqrt(np.sum((x_ratings - y_ratings)**2)))

def manhattanDistance(df, user_x, user_y):
    corated_items = getCoRatedItems(df, user_x, user_y)

    if corated_items.empty:
        return 0

    x_ratings = corated_items['rating_x']
    y_ratings = corated_items['rating_y']

    return 1 / (1 + np.sum(np.abs(x_ratings - y_ratings)))

def getNeighbors(df, user, item = None, blacklist = [], sim_fun = pearsonSimilarity, sim_th = 0.6, k = 30, overlap_th = 10):
    neighbors = []

    candidates = df['userId'].unique()
    candidates = pd.DataFrame({'userId': candidates})
    candidates = candidates[(candidates['userId'] != user) & (~candidates['userId'].isin(blacklist))]
    candidates = candidates['userId'].values
    
    for candidate in candidates:
        corated_items_num = getCoRatedItems(df, user, candidate).shape[0]
        
        if corated_items_num >= overlap_th:
            sim = sim_fun(df, user, candidate)
            if abs(sim) > sim_th:
                if item is None or not df[(df['userId'] == candidate) & (df['movieId'] == item)].empty:
                    neighbors.append((candidate, sim))
                    if sim_th != 0 and len(neighbors) == k:
                        break

    return sorted(neighbors, key=lambda x: abs(x[1]), reverse=True)[:k]

def customGetNeighbors(df, user, item, blacklist = [], sim_th1 = 0.6, sim_th2 = 0.4, sim_fun = pearsonSimilarity, k1 = 5, k2 = 15):
    neighbors = []

    sim_based_neighbors = getNeighbors(df, user, None, blacklist, sim_fun, sim_th1, k1)
    neighbors.extend(sim_based_neighbors)

    item_based_neighbors = getNeighbors(df, user, item, blacklist, sim_fun, sim_th2, k2)
    neighbors.extend(item_based_neighbors)
    
    return neighbors

def basePred(df, user, item, sim_th = 0.3, sim_fun = pearsonSimilarity):
    neighbors = getNeighbors(df, user, item, [], sim_fun, sim_th)
    u_mean = np.mean(df[df['userId'] == user]['rating'])
    num, den = 0, 0

    for neighbor, sim in neighbors:
        n_ratings = df[(df['userId'] == neighbor) & (df['movieId'] == item)]['rating']

        if not n_ratings.empty:
            n_mean = np.mean(df[df['userId'] == neighbor]['rating'])
            num += sim * (n_ratings.values[0] - n_mean)
            den += abs(sim)

    if den == 0 or neighbors == []:
        return 0

    return u_mean + num / den

def recursivePred(df, user, item, lev = 0, blacklist = [], sim_th1 = 0.6, sim_th2 = 0.4, sim_fun = pearsonSimilarity, lmb = 0.5, lev_th = 1):
    if lev >= lev_th:
        return basePred(df, user, item, sim_th2, sim_fun)

    neighbors = customGetNeighbors(df, user, item, blacklist, sim_th1, sim_th2, sim_fun)
    u_mean = np.mean(df[df['userId'] == user]['rating'])
    num, den = 0, 0
    
    for neighbor, sim in neighbors:
        n_ratings = df[(df['userId'] == neighbor) & (df['movieId'] == item)]['rating']
        n_mean = np.mean(df[df['userId'] == neighbor]['rating'])

        if not n_ratings.empty:
            num += sim * (n_ratings.values[0] - n_mean)
            den += abs(sim)
        else:
            blacklist.append(user)
            num += lmb * sim * (recursivePred(df, neighbor, item, lev+1, blacklist.copy(), sim_th1, sim_th2, sim_fun, lmb, lev_th) - n_mean)
            den += lmb * abs(sim)            

    if den == 0 or neighbors == []:
        return 0
    else:
        return u_mean + num / den
    
def getRecommendedItems(df, user, rate_th = 4.5, pred_th = 4, k1 = 10, k2 = 10, sim_fun = pearsonSimilarity):
    u_items = set(df[df['userId'] == user]['movieId'])
    
    neighbors = getNeighbors(df, user, None, [], sim_fun, 0.6, k1)
    
    n_items = set()
    for neighbor, _ in neighbors:
        items = set(df[(df['userId'] == neighbor) & (df['rating'] >= rate_th)]['movieId'])
        n_items.update(items)

    not_yet_rated = list(n_items - u_items)
    random.shuffle(not_yet_rated)

    item_to_pred = []

    with pb.ProgressBar(max_value=k2) as bar:
        for item in not_yet_rated:
            pred = basePred(df, user, item)

            if pred >= pred_th:
                item_to_pred.append((item, pred))
                bar.next()
            if pred_th != 0 and len(item_to_pred) == k2:
                break

    return sorted(item_to_pred, key=lambda x: x[1], reverse=True)[:k2]

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

def evaluatePred(df, sample, pred_fun = recursivePred, sim_th1 = 0.6, sim_th2 = 0.4, sim_fun = pearsonSimilarity, lmb = 0.5, lev_th = 1):
    err = []
    tot_time = 0

    with pb.ProgressBar(max_value=len(sample)) as bar:
        for user, item in sample:
            real_pred = df[(df['userId'] == user) & (df['movieId'] == item)]['rating'].values[0]

            if pred_fun.__name__ == 'basePred':
                start_time = time.time()
                generated_pred = pred_fun(df, user, item, sim_th2, sim_fun)
                end_time = time.time()
                tot_time += end_time - start_time
            else:
                start_time = time.time()
                generated_pred = pred_fun(df, user, item, 0, [], sim_th1, sim_th2, sim_fun, lmb, lev_th)
                end_time = time.time()
                tot_time += end_time - start_time

            err.append(abs(generated_pred - real_pred))
            bar.next()

    return np.mean(err), np.std(err), np.max(err), tot_time/len(sample)

def main():
    df_path = os.path.join(os.getcwd(), 'group_recommendations', 'ml-latest-small', 'ratings.csv')
    df = pd.read_csv(df_path)

    sample = getRandomSample(df, 5, 10)

    headers = ['pred_fun', 'sim_fun', 'sim_th1', 'sim_th2', 'lmb', 'lev_th', 'mean_err', 'std_err', 'max_err', 'mean_time']

    # Evaluating sim_th2 on basePred
    stats = pd.DataFrame(columns=headers)
    for sim_th2 in np.arange(0, 0.6, 0.1):
        mean_err, std_err, max_err, mean_time = evaluatePred(df, sample, basePred, None, sim_th2)
        new_row = ['basePred', 'pearsonSimilarity', np.nan, sim_th2, np.nan, np.nan, mean_err, std_err, max_err, mean_time]
        stats.loc[len(stats)] = new_row

    print(stats)
    stats.to_csv('sim_th2.csv', index=False)

    # Evaluating similarities on basePred
    stats = pd.DataFrame(columns=headers)
    SIM_TH2= 0.1
    for sim_fun in [pearsonSimilarity, cosineSimilarity, jaccardSimilarity, euclideanDistance, manhattanDistance]:
        mean_err, std_err, max_err, mean_time = evaluatePred(df, sample, basePred, None, SIM_TH2, sim_fun)
        new_row = ['basePred', sim_fun.__name__, np.nan, SIM_TH2, np.nan, np.nan, mean_err, std_err, max_err, mean_time]
        stats.loc[len(stats)] = new_row

    print(stats)
    stats.to_csv('similarities.csv', index=False)

    # Evaluating sim_th1 on recursivePred
    stats = pd.DataFrame(columns=headers)
    SIM_TH2, LMB, LEV_TH = 0.1, 1, 1
    for sim_th1 in np.arange(0, 0.8, 0.1):
        mean_err, std_err, max_err, mean_time = evaluatePred(df, sample, recursivePred, sim_th1, SIM_TH2, pearsonSimilarity, LMB, LEV_TH)
        new_row = ['recursivePred', 'pearsonSimilarity', sim_th1, SIM_TH2, LMB, LEV_TH, mean_err, std_err, max_err, mean_time]
        stats.loc[len(stats)] = new_row

    print(stats)
    stats.to_csv('sim_th1.csv', index=False)

    # Evaluating lmb on recursivePred
    stats = pd.DataFrame(columns=headers)
    SIM_TH1, SIM_TH2, LEV_TH = 0.5, 0.1, 1
    for lmb in np.arange(0, 1.1, 0.1):
        mean_err, std_err, max_err, mean_time = evaluatePred(df, sample, recursivePred, SIM_TH1, SIM_TH2, pearsonSimilarity, lmb, LEV_TH)
        new_row = ['recursivePred', 'pearsonSimilarity', SIM_TH1, SIM_TH2, lmb, LEV_TH, mean_err, std_err, max_err, mean_time]
        stats.loc[len(stats)] = new_row

    print(stats)
    stats.to_csv('lmb.csv', index=False)

    # Evaluating lev_th on recursivePred
    stats = pd.DataFrame(columns=headers)
    SIM_TH1, SIM_TH2, LMB= 0.5, 0.1, 0.8
    for lev_th in np.arange(0, 4, 1):
        mean_err, std_err, max_err, mean_time = evaluatePred(df, sample, recursivePred, SIM_TH1, SIM_TH2, pearsonSimilarity, LMB, lev_th)
        new_row = ['recursivePred', 'pearsonSimilarity', SIM_TH1, SIM_TH2, LMB, lev_th, mean_err, std_err, max_err, mean_time]
        stats.loc[len(stats)] = new_row

    print(stats)
    stats.to_csv('lev_th.csv', index=False)


if __name__ == "__main__":
    main()