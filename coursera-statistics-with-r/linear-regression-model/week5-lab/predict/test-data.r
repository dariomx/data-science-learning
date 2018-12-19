
# american sniper
m1 = data.frame(audience_score=84, runtime=134, critics_score=72,
                genre="Drama", title_type="Feature Film",
                imdb_rating=7.3, imdb_num_votes=388512,
                best_pic_nom="no")

# revenant
m2 = data.frame(audience_score=84, runtime=156, critics_score=78,
                genre="Drama", title_type="Feature Film",
                imdb_rating=8.0, imdb_num_votes=585140,
                best_pic_nom="yes")

test.data = rbind(m1, m2)
