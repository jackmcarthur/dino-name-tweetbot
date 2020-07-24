from tensorflow import keras
import tweepy
import schedule
import time
import numpy as np
import csv
import editdistance

with open('dinonames.csv', newline='') as f:
    reader = csv.reader(f)
    importedlist = list(reader)
    dinonames = [val for sublist in importedlist for val in sublist][1:]
    
text = ''.join(dinonames)
vocab = sorted(set(text))

# Creating a mapping from unique characters to integers
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
max_char = len(max(dinonames, key = len))
char_dim = len(vocab)

model = keras.models.load_model('dino_model')


# a name generation function used in its simplest form for the training below
# it can generate a name with or without a starting character start_char,
# todo: implement a temperature that controls how predictable the text is
def generate_name(model, start_char=None, temp=1.0):
    name = []
    x = np.zeros((1, max_char, char_dim))
    i = 0
    end = False

    if start_char is not None:
      start_char = start_char.lower()
      x[0,0,char2idx[start_char]] = 1
      name.append(start_char)
      i = 1

    while end == False:
        probs = list(model.predict(x)[0,i])
        norm_probs = probs / np.sum(probs)
        index = np.random.choice(range(char_dim), p=norm_probs)
        if i == max_char-2:
            char = '\n'
            end = True
        else:
            char = idx2char[index]
        name.append(char)
        x[0, i+1, index] = 1
        i += 1
        if char == '\n':
            end = True
    
    name = ''.join(name)
    return name

# function that creates a dinosaur name with a minimum Levenshtein distance from an existing name
# with min_dist = 2, "Dipladocus" and "Dilodocus" both get rejected for being 1 string edit from "Diplodocus"
def checked_name(model, start_char=None, temp=1.0, min_dist=2):
    tries = 1
    while 1 == 1:
        name = generate_name(model, start_char=None, temp=1.0)
        if name != '\n':
            if min([editdistance.eval(dino, name) for dino in dinonames]) >= min_dist:
                break
            else:
                tries += 1
    return name.capitalize(), tries

# Authenticate to Twitter (this can be done by someone with a developer account)
import os
api_key = os.environ.get("api_key", None)
api_key_secret = os.environ.get("api_key_secret", None)
access_token = os.environ.get("access_token", None)
access_token_secret = os.environ.get("access_token_secret", None)

auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

# Create API object
api = tweepy.API(auth)

tweets = 0
def tweet_check():
	global tweets
	tweet = checked_name(model)[0]
	# write a tweet
	api.update_status(tweet)
	print(tweet)
	tweets += 1

schedule.every(30).seconds.do(tweet_check)

if __name__ == '__main__':
	# This would be how it constantly runs the check
	while True:
		if tweets >= 50:
			break
		schedule.run_pending()
		time.sleep(1)

