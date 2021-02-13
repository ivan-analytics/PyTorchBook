from .training.ReviewClassifier import ReviewClassifier
from .data_managing.Dataset import ReviewDataset
from .training.hyperparameters import args

from .testing import compute_loss_acc as loss_acc
from .testing import predict_rating as predict
from .testing import analizing as analyze

if __name__ == '__main__':
    # creating dataset
    if args.reload_from_files:
        # training from a checkpoint
        print("Loading dataset and vectorizer")
        dataset = ReviewDataset.load_dataset_and_load_vectorizer(args.review_csv,
                                                                 args.vectorizer_file)
    else:
        print("Loading dataset and creating vectorizer")
        # create dataset and vectorizer
        dataset = ReviewDataset.load_dataset_and_make_vectorizer(args.review_csv)
        dataset.save_vectorizer(args.vectorizer_file)
    vectorizer = dataset.get_vectorizer()
    classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab))

    # computing loass and accuracy of model
    loss_acc.compute(classifier, args, dataset)

    # testing model on real data
    test_review = "не бери грех на душу"
    classifier = classifier.cpu()
    prediction = predict.predict_rating(test_review, classifier, vectorizer, decision_threshold=0.5)
    print("{} -> {}".format(test_review, prediction))

    analyze.influencial_words(classifier, vectorizer)
