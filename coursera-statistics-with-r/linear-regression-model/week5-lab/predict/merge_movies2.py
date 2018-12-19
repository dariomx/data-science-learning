import csv

rating_file = "movie_ratings_16_17.csv"
test_data_file = "test-data.csv"

def norm_title(title):
    return " ".join(title.lower().split())

with open(rating_file, "r") as fin, open(test_data_file, "w") as fout:
    reader = csv.reader(fin, delimiter=',', quotechar='"')
    writer = csv.writer(fout, delimiter=',', quotechar='"')
    next(reader)
    out_row = ["title", "imdb_rating", "critics_score", "audience_score"]
    writer.writerow(out_row)
    for in_row in reader:
        try:
            title = norm_title(in_row[0])
            imdb_score = in_row[3]
            critics_score = in_row[4]
            audience_score = in_row[5]
            out_row = [title, imdb_score, critics_score, audience_score]
            writer.writerow(out_row)
        except:
            print("error parsing line: " + " ".join(in_row))
            raise

