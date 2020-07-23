# dino-name-twitterbot
A python project which uses data scraped from Wikipedia's List of Dinosaur Genera to train a neural network that generates new dinosaur names. The new names are then tweeted by a bot using the Twitter API.

## Demonstration

| Real Dinosaurs | Fake Dinosaurs |
|----------------|----------------|
| Epicampodon    | Achinops       |
| Clarencea      | Rhaeosaurus    |
| Lusovenator    | Iscrodromeus   |
| Lufengosaurus  | Unesasaurus    |
| Huanghetitan   | Ormalong       |
| Albertavenator | Ugansaurus     |
| Masiakasaurus  | Arshansaurus   |
| Adeopapposaurus| Nenongo        |
| Beishanlong    | Haplodon       |
| Yamanasaurus   | Erisanosaurus  |

## Network design and implementation
Tensorflow and Keras were then used to create a small character-based recurrent neural network, a type of neural network commonly used for short-string text generation. The neural network was trained with 1648 dinosaur names scraped from the [Wikipedia list of dinosaur genera](https://en.wikipedia.org/wiki/List_of_dinosaur_genera) using the BeautifulSoup package (saved to [dinonames.csv](dinonames.csv)), allowing it to produce realistic dinosaur names of its own. The generated names are then filtered using the stringdistance package, ensuring that all outputs are at least two single-character edits away from any real dinosaur names (dinosaur trademark law is not to be messed with), and they are finally posted to Twitter using the Tweepy package.

The network itself is stored in [dino_model.h5](/dino_model.h5). It can easily be loaded into Keras, and the following code is all that is necessary to interact with the model (though a more complete scaffolding code can be found in the notebook [dino-bot.ipynb](/dino-bot.ipynb)
```python
from tensorflow import keras
model = keras.models.load_model('dino_model.h5')

x = np.zeros((1, max_char, char_dim))
model.predict(x)[0,i]
```

## Naming function
The function `dino-bot.checked_name()` returns a tuple `name, attempts` of the name generated using [dino_model.h5](/dino_model.h5), and the number of attempts required to obtain a name with an acceptably high stringdistance `min_dist` from all existing dinosaur names in [dinonames.csv](dinonames.csv). By default, `min_dist` is 2, so that "Dipladocus" and "Dilodocus" both get rejected for being 1 string edit from "Diplodocus."

```python
for i in range(10):
    print(checked_name(model)[0],end='')
```
```
Euisabia
Ultacephodon
Inkissaceus
Rabiatitan
Edmorosaurus
Unbyrosaurus
Arshanglangosaurus
Erbosaurus
Sindasaurus
Epidemtrus
```
The number of attempts to generate a name increases significantly as `min_dist` is increased.
```python
for i in range(4):
    num_attempts = [checked_name(model, min_dist = i)[1] for j in range(20)]
    print('min_dist =', min_dist, ', average number of attempts = ', sum(num_attempts)/len(num_attempts))
```
```
min_dist = 1 , average number of attempts =  1.0
min_dist = 2 , average number of attempts =  1.3
min_dist = 3 , average number of attempts =  1.85
min_dist = 4 , average number of attempts =  2.05
```
Names can also be generated that begin with a character given by passing the optional argument `start_char`:
```python
for i in range(5):
    print(checked_name(model, start_char='g')[0],end='')
```
```
Gyoodon
Gyongosaurus
Gaviraptor
Gdilosaurus
Gyonosaurus
```

This neural network project pulls together web scraping, machine learning, file writing, and API interfacing in Python, making it an excellent programming exercise and an elegant showcase of many of the practical abilities I've developed. In the future I hope to add temperature controls to this neural network to vary the predictability of the model's output, and eventually add more features to the Twitter posts, such as an image or theorized country of origin.
