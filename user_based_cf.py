import numpy as np
import pandas as pd
import random
import os

def getCoRatedItems(df, userX, userY):
    xRatings = df[df['userId'] == userX]
    yRatings = df[df['userId'] == userY]

    return pd.merge(xRatings, yRatings, on='movieId', how='inner')

def pearsonSimilarity(df, userX, userY):
    coRatedItems = getCoRatedItems(df, userX, userY)

    if coRatedItems.empty:
        return 0

    xRating = coRatedItems['rating_x']
    yRating = coRatedItems['rating_y']

    xMean = np.mean(xRating)
    yMean = np.mean(yRating)

    den = np.sqrt(np.sum((xRating - xMean)**2)) * np.sqrt(np.sum((yRating - yMean)**2))
    if den == 0:
        return 0
    else:
        return np.sum((xRating - xMean) * (yRating - yMean)) / den
    
def cosineSimilarity(df, userX, userY):
    coRatedItems = getCoRatedItems(df, userX, userY)

    if coRatedItems.empty:
        return 0

    xRating = coRatedItems['rating_x']
    yRating = coRatedItems['rating_y']

    ratingMean = np.mean(df[df['movieId'].isin(coRatedItems['movieId'])]['rating'])
    
    num = np.dot(xRating - ratingMean, yRating - ratingMean)
    den = np.linalg.norm(xRating - ratingMean) * np.linalg.norm(yRating - ratingMean)

    if den == 0:
        return 0
    else:
        return num / den
    
def triangleSimilarity(df, userX, userY):
    coRatedItems = getCoRatedItems(df, userX, userY)

    if coRatedItems.empty:
        return 0

    xRating = coRatedItems['rating_x']
    yRating = coRatedItems['rating_y']

    xMean = np.mean(xRating)
    yMean = np.mean(yRating)

    num = np.sqrt(np.sum((xRating - yRating)**2))
    den = np.sqrt(np.sum((xRating)**2)) + (np.sqrt(np.sum((yRating)**2)))
    triangle = 1 - num / den

    absMeanErr = abs(xMean - yMean)
    absStdErr = abs(xRating.std(ddof=0) - yRating.std(ddof=0))
    userRatingPref = 1 - 1 / (1 +  np.exp(- absMeanErr * absStdErr))

    return triangle * userRatingPref

def jaccardSimilarity(df, userX, userY):
    coRatedItems = getCoRatedItems(df, userX, userY)

    if coRatedItems.empty:
        return 0

    xRating = set(coRatedItems['rating_x'])
    yRating = set(coRatedItems['rating_y'])

    intersection = len(xRating.intersection(yRating))
    union = len(xRating.union(yRating))

    if union == 0:
        return 0
    return intersection / union

def euclideanDistance(df, userX, userY):
    coRatedItems = getCoRatedItems(df, userX, userY)

    if coRatedItems.empty:
        return 0

    xRating = coRatedItems['rating_x']
    yRating = coRatedItems['rating_y']

    return 1 / (1 + np.sqrt(np.sum((xRating - yRating)**2)))

def manhattanDistance(df, userX, userY):
    coRatedItems = getCoRatedItems(df, userX, userY)

    if coRatedItems.empty:
        return 0

    xRating = coRatedItems['rating_x']
    yRating = coRatedItems['rating_y']

    return 1 / (1 + np.sum(np.abs(xRating - yRating)))

def getNeighbors(df, user, item = None, blacklist = [], similarityFun = pearsonSimilarity, simTh = 0.6, k = 30, overlapTh = 10):
    neighbors = []

    for candidate in df['userId'].unique():
        coRatedItemsNum = getCoRatedItems(df, user, candidate).shape[0]
        
        if user != candidate and candidate not in blacklist and coRatedItemsNum >= overlapTh:
            sim = similarityFun(df, user, candidate)
            if item == None:
                if abs(sim) > simTh:
                    neighbors.append((candidate, sim))
                    if simTh != 0 and len(neighbors) == k:
                        break
            else:
                if not df[(df['userId'] == candidate) & (df['movieId'] == item)].empty:
                    if abs(sim) > simTh:
                        neighbors.append((candidate, sim))
                        if simTh != 0 and len(neighbors) == k:
                            break

    return sorted(neighbors, key=lambda x: abs(x[1]), reverse=True)[:k]

def customGetNeighbors(df, user, item, blacklist = [], similarityFun = pearsonSimilarity, k1 = 5, k2 = 15):
    neighbors = []

    simBasedNeighbors = getNeighbors(df, user, None, blacklist, similarityFun, 0.6, k1)
    neighbors.extend(simBasedNeighbors)

    itemBasedNeighbors = getNeighbors(df, user, item, blacklist, similarityFun, 0.4, k2)
    neighbors.extend(itemBasedNeighbors)
    
    return neighbors

def basePred(df, user, item, similarityFun = pearsonSimilarity):
    neighbors = getNeighbors(df, user, item, [], similarityFun, 0.3)
    uMean = np.mean(df[df['userId'] == user]['rating'])
    num, den = 0, 0

    for neighbor, sim in neighbors:
        nRating = df[(df['userId'] == neighbor) & (df['movieId'] == item)]['rating']

        if not nRating.empty:
            nMean = np.mean(df[df['userId'] == neighbor]['rating'])
            num += sim * (nRating.values[0] - nMean)
            den += abs(sim)

    if den == 0 or neighbors == []:
        return 0

    return uMean + num / den

def recursivePred(df, user, item, lev = 0, blacklist = [], similarityFun = pearsonSimilarity, lmb = 0.5, levTh = 1):
    if lev >= levTh:
        return basePred(df, user, item)

    neighbors = customGetNeighbors(df, user, item, blacklist, similarityFun)
    uMean = np.mean(df[df['userId'] == user]['rating'])
    num, den = 0, 0
    
    for neighbor, sim in neighbors:
        nRating = df[(df['userId'] == neighbor) & (df['movieId'] == item)]['rating']
        nMean = np.mean(df[df['userId'] == neighbor]['rating'])

        if not nRating.empty:
            num += sim * (nRating.values[0] - nMean)
            den += abs(sim)
        else:
            blacklist.append(user)
            num += lmb * sim * (recursivePred(df, neighbor, item, lev+1, blacklist.copy()) - nMean)
            den += lmb * abs(sim)            

    if den == 0 or neighbors == []:
        return 0
    else:
        return uMean + num / den
    
def getRecommendedItems(df, user, rateTh = 4.5, predTh = 4, k1 = 10, k2 = 10, similarityFun=pearsonSimilarity):
    uItems = set(df[df['userId'] == user]['movieId'])
    
    neighbors = getNeighbors(df, user, None, [], similarityFun, 0.6, k1)
    
    nItems = set()
    for neighbor, _ in neighbors:
        tmp = set(df[(df['userId'] == neighbor) & (df['rating'] >= rateTh)]['movieId'])
        nItems.update(tmp)

    print(len(nItems))
    notYetRated = list(nItems - uItems)
    random.shuffle(notYetRated)

    item2Pred = []
    for item in notYetRated:
        pred = basePred(df, user, item)

        if pred >= predTh:
            item2Pred.append((item, pred))
        if predTh != 0 and len(item2Pred) == k2:
            break

    return sorted(item2Pred, key=lambda x: x[1], reverse=True)[:k2]

def evaluateSimilarity(df, user):
    uItems = list(df[df['userId'] == user]['movieId'])[:5]

    pearsonPed, cosinePred, trianglePred, jaccardPred, euclideanPred, manhattanPred  = [], [], [], [], [], [] 

    for item in uItems:
        itemPred = df[(df['userId'] == user) & (df['movieId'] == item)]['rating'].values[0]

        pearsonPed.append(abs(basePred(df, user, item, pearsonSimilarity) - itemPred))
        cosinePred.append(abs(basePred(df, user, item, cosineSimilarity) - itemPred))
        trianglePred.append(abs(basePred(df, user, item, triangleSimilarity) - itemPred))
        jaccardPred.append(abs(basePred(df, user, item, jaccardSimilarity) - itemPred))
        euclideanPred.append(abs(basePred(df, user, item, euclideanDistance) - itemPred))
        manhattanPred.append(abs(basePred(df, user, item, manhattanDistance) - itemPred))

    print("Pearson mean err: ", np.mean(pearsonPed))
    print("Cosine mean err: ", np.mean(cosinePred))
    print("Triangle mean err: ", np.mean(trianglePred))
    print("Jaccard mean err: ", np.mean(jaccardPred))
    print("Euclidean mean err: ", np.mean(euclideanPred))
    print("Manhattan mean err: ", np.mean(manhattanPred))

def evaluatePrediction(df, user):
    uItems = list(df[df['userId'] == user]['movieId'])[:50]

    standardPred, recPred  = [], []

    for item in uItems:
        itemPred = df[(df['userId'] == user) & (df['movieId'] == item)]['rating'].values[0]

        standardPred.append(abs(basePred(df, user, item) - itemPred))
        recPred.append(abs(recursivePred(df, user, item) - itemPred))

    print("Standard prediction mean err: ", np.mean(standardPred))
    print("Recursive prediction mean err: ", np.mean(recPred))

def main():
    dfPath = os.path.join(os.getcwd(), 'group_recommendations', 'ml-latest-small', 'ratings.csv')
    df = pd.read_csv(dfPath)
    print(df.head())
    print("\nRow num: ", df.shape[0])

    user = 1  
    #print(getRecommendedItems(df, user))
    evaluatePrediction(df, 1)

if __name__ == "__main__":
    main()