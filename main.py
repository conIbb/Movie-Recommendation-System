import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


movieData = pd.read_csv("movies.csv")

movieData['genres'] = movieData['genres'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')

tfidfMatrix = tfidf.fit_transform(movieData['genres'])

cosineSim = linear_kernel(tfidfMatrix,tfidfMatrix)

indices = pd.Series(movieData.index, index=movieData['title']).drop_duplicates()

def getRecommendations(title, cosineSim=cosineSim):
    idx = indices[title]

    simScore = list(enumerate(cosineSim[idx]))

    simScore = sorted(simScore, key=lambda x: x[1], reverse = True)

    simScore = simScore[1:11]

    movieIndices = [i[0] for i in simScore]
    
    return movieData['title'].iloc[movieIndices]
print(getRecommendations('Toy Story (1995)'))
