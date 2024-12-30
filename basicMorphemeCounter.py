import morfessor
import pickle


corpus = []
with open('mc.txt', 'r') as file: # mc.txt is a preprocessed corpus containing words, bigrams, and trigrams
    for line in file:
        word = line.strip() # each word/bigram/trigram is on a new line
        corpus.append((1.0, word))


corpus_iterator = iter(corpus)
model = morfessor.BaselineModel()
model.train_online(corpus_iterator)


with open('trained_model.pkl', 'wb') as f:
    pickle.dump(model, f) # save trained model

with open('trained_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f) # load trained model

def count_morphemes(word):
    try:
        segmentation = loaded_model.segment(word)
        return len(segmentation)
    except KeyError:
        print(f"Warning: Word '{word}' not found in model analysis.")
        return 'NULL'

# example usage
words = ["cats", "determined", "unhappiness", "cardiogram", "activity", 'geriatric', 'unkind']
morpheme_counts = {word: count_morphemes(word) for word in words}
print(morpheme_counts)
