import torch

def influencial_words(classifier, vectorizer):
    # Сортировка весов
    fc1_weights = classifier.fc1.weight.detach()[0]
    _, indices = torch.sort(fc1_weights, dim=0, descending=True)
    indices = indices.numpy().tolist()
    # Топ-20 позитивных слов
    print("Influential words in Positive Reviews:")
    print("--------------------------------------")
    for i in range(20):
        print(vectorizer.review_vocab.lookup_index(indices[i]))

    print("Influential words in Negative Reviews:")
    print("--------------------------------------")
    indices.reverse()
    for i in range(20):
        print(vectorizer.review_vocab.lookup_index(indices[i]))