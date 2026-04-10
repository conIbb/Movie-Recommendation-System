import pandas as pd

movieData = pd.read_csv("movies.csv")
ratingsData = pd.read_csv("ratings.csv")

print(movieData.head)
print(ratingsData.head)