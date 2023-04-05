import pickle

label_set = {"B-PARTY", "I-PARTY", "O"}

with open("label_set.pkl", "wb") as fp:
    pickle.dump(label_set, fp)
