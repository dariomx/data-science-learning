import csv

meta_file = "movie_metadata.csv"
rating_file = "movie_ratings_16_17.csv"
actors_file = "../movies.csv"

def norm_title(title):
    return " ".join(title.lower().split())

movie = dict()
with open(rating_file, "r") as fin:
    reader = csv.reader(fin, delimiter=',', quotechar='"')
    next(reader)
    for row in reader:
        try:
            title = norm_title(row[0])
            year = int(row[1])
            score = int(row[4])
            movie[title] = [year, score]
        except:
            print("error parsing line: " + " ".join(row))
            raise

with open(meta_file, "r") as fin:
    reader = csv.reader(fin, delimiter=',', quotechar='"')
    next(reader)
    for row in reader:
        try:
            title = norm_title(row[11])
            act1 = row[10]
            act2 = row[6]
            act3 = row[14]
            if title in movie:
                movie[title] += [act1, act2, act3]
            else:
                print("no metadata for: " + title)
        except:
            print("error parsing line: " + " ".join(row))
            raise

actor1 = set()
actor2 = set()
actor3 = set()
with open(actors_file, "r") as fin:
    reader = csv.reader(fin, delimiter=',', quotechar='"')
    next(reader)
    for row in reader:
        try:
            actor1.add(row[25])
            actor2.add(row[26])
            actor3.add(row[27])
        except:
            print("error parsing line: " + " ".join(row))
            raise

def good_movie(md):
    return len(md) == 5 and \
        md[2] in actor1 

movie = {m:md for (m,md) in movie.items() if good_movie(md)}
print(len(movie))
print(movie)

