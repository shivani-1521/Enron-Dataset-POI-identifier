#!/usr/bin/python

def dataset_exploration(enron_data):
    print "The total number of data points available are:", len(enron_data)

    max_features = 0
    for person in enron_data:
        features = len(enron_data[person])
        if features > max_features:
            max_features = features

    print "Each person has a maximum of " + str(max_features) + " features"

    total_poi = 0
    for item in enron_data:
        if enron_data[item]["poi"] == 1:
            total_poi += 1

    print "The number of POIs in the dataset is", total_poi

    with open("poi_names.txt") as f:
        names = f.read().splitlines()
        print "All the POIs are", len(names) - 2

def parse_labels(label):
    """
    Pass in a label and return it capitalised and without underscores
    :param label: str
    :return: str
    """
    words = label.split("_")
    parsed_label = " ".join(words).capitalize()
    return parsed_label


def visualise_features(feature1, feature2, fig_name):
    """
    Make a scatterplot of two features and save the figure
    :param feature1: str
    :param feature2: str
    :param fig_name: str
    :return: None
    """
    for person in data_dict:
        for feature in data_dict[person]:
            if feature == feature1:
                x = data_dict[person][feature]
            if feature == feature2:
                y = data_dict[person][feature]
        plt.scatter(x = x, y = y, color = "#FF6666")
    plt.title(fig_name)
    plt.xlabel(parse_labels(feature1))
    plt.ylabel(parse_labels(feature2))
    plt.savefig(fig_name + ".png")
    plt.close()


def compute_fraction(poi_messages, all_messages):
    """
    Compute the fraction of messages send and received from a POI
    :param poi_messages: int
    :param all_messages: int
    :return: float
    """
    if poi_messages == "NaN" or all_messages == "NaN":
        fraction = 0
    else:
        fraction = float(poi_messages) / all_messages
    return fraction


def create_features():
    """
    Creates 2 new features: "fraction_from_poi" and "fraction_to_poi"
    :return: None
    """
    for name in data_dict:

        data_point = data_dict[name]

        from_poi_to_this_person = data_point["from_poi_to_this_person"]
        to_messages = data_point["to_messages"]
        fraction_from_poi = compute_fraction( from_poi_to_this_person, to_messages )
        data_point["fraction_from_poi"] = fraction_from_poi

        from_this_person_to_poi = data_point["from_this_person_to_poi"]
        from_messages = data_point["from_messages"]
        fraction_to_poi = compute_fraction( from_this_person_to_poi, from_messages )
        data_point["fraction_to_poi"] = fraction_to_poi

        data_dict[name] = data_point


if __name__ == "__main__":

    import pickle
    import matplotlib.pyplot as plt
    import seaborn as sns
    from feature_format import featureFormat, targetFeatureSplit
    from tester import dump_classifier_and_data

    # Load the dictionary containing the data set
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)

    dataset_exploration(data_dict)

    # Visualise "salary" and "bonus" features to identify outliers
    visualise_features("salary", "bonus", "Salary vs. Bonus with Outliers")

    # Remove outliers
    data_dict.pop("TOTAL", 0)
    data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

    # Visualise "salary" and "bonus" again without outliers
    visualise_features("salary", "bonus", "Salary vs. Bonus without Outliers")

    # Create new features of "fraction_from_poi" and "fraction_to_poi"
    create_features()

    # Visualise the new features
    visualise_features("fraction_from_poi", "fraction_to_poi", "Fraction from POI vs. Fraction to POI")

    # Store to my_dataset for easy export below.
    my_dataset = data_dict

    # Select features to use
    features_list = ["poi", "salary", "fraction_from_poi", "fraction_to_poi", "deferral_payments", "total_payments", \
                     "loan_advances", "bonus", "restricted_stock_deferred", "deferred_income", "total_stock_value", \
                     "expenses", "exercised_stock_options", "other", "long_term_incentive", "restricted_stock", \
                     "director_fees", "to_messages", "from_poi_to_this_person", "from_messages", \
                     "from_this_person_to_poi", "shared_receipt_with_poi"]

    # Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.cross_validation import StratifiedShuffleSplit
    from sklearn.grid_search import GridSearchCV

    selector = SelectKBest(f_classif)

    nb = GaussianNB()

    dt = DecisionTreeClassifier()

    pipeline1 = Pipeline([("kbest", selector), ("nb", nb )])

    pipeline2 = Pipeline([("kbest", selector), ("dt", dt )])

    params1 = {"kbest__k": range(5, 15)}

    params2 = {"kbest__k": range(5, 15),
               "dt__min_samples_split": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
               "dt__min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
               "dt__criterion": ["gini", "entropy"]}

    sss = StratifiedShuffleSplit(labels, 100, test_size=0.3, random_state=60)

    gs = GridSearchCV(pipeline1, params1, n_jobs=-1, cv=sss)
    gs.fit(features, labels)

    clf1 = gs.best_estimator_.named_steps["nb"]

    from tester import test_classifier

    print "Tester Classification report"
    test_classifier(clf1, data_dict, features_list)

    features_used = gs.best_estimator_.named_steps["kbest"].get_support(indices=True)
    print "A total of %d features were used" % len(features_used)
    feature_names = [features_list[i + 1] for i in features_used]
    print "The features used are:", feature_names

    gs = GridSearchCV(pipeline2, params2, n_jobs=-1, cv=sss, scoring="f1")
    gs.fit(features, labels)

    clf2 = gs.best_estimator_.named_steps["dt"]

    print "Tester Classification report"
    test_classifier(clf2, data_dict, features_list)

    features_used = gs.best_estimator_.named_steps["kbest"].get_support(indices=True)
    print "A total of %d features were used" % len(features_used)
    feature_names = [features_list[i + 1] for i in features_used]
    print "The features used are:", feature_names

    # Dump the classifier, the dataset and the feature list in pickle files for easy evaluation by tester.py
    dump_classifier_and_data(clf2, my_dataset, features_list)