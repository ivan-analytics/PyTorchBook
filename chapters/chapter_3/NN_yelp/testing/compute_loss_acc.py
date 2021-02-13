import torch
from ..HelperUtilities.training_state import make_train_state, compute_accuracy
from ..data_managing.Dataset import generate_batches


"""
sample of using function

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

    loss_acc.compute(classifier, args, dataset)
"""

# compute the loss & accuracy on the test set using the best available model
def compute(classifier, args, dataset):
    classifier = classifier.to(args.device)
    train_state = make_train_state(args)

    # load model parameters from file
    model_file_path = "NN_yelp/model_storage/ch3/yelp/" + train_state['model_filename']
    classifier.load_state_dict(torch.load(model_file_path))
    classifier = classifier.to(args.device)

    dataset.set_split('test')
    batch_generator = generate_batches(dataset,
                                       batch_size=args.batch_size,
                                       device=args.device)
    running_loss = 0.
    running_acc = 0.
    classifier.eval()

    for batch_index, batch_dict in enumerate(batch_generator):
        # compute the output
        y_pred = classifier(x_in=batch_dict['x_data'].float())

        # compute the loss
        loss_func = torch.nn.BCEWithLogitsLoss()
        loss = loss_func(y_pred, batch_dict['y_target'].float())
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)

        # compute the accuracy
        acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
        running_acc += (acc_t - running_acc) / (batch_index + 1)

    train_state['test_loss'] = running_loss
    train_state['test_acc'] = running_acc

    print("Test loss: {:.3f}".format(train_state['test_loss']))
    print("Test Accuracy: {:.2f}".format(train_state['test_acc']))
