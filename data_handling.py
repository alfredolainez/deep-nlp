import json
import pickle
import random
import sys

import language

DEFAULT_REVIEWS_FILE = "data/yelp_academic_dataset_review.json"
DEFAULT_REVIEWS_PICKLE = "data/reviews.pickle"

def pickles_from_json(json_file=DEFAULT_REVIEWS_FILE, pickle_name=DEFAULT_REVIEWS_PICKLE, num_partitions=100,
                      accepted=None):
    """
    Dumps a json into a number of pickle partitions, which contain a list of python objects.

    accepted is a generic function that returns true or false for a single json object, specifying whether or not
    the object should be added to the pickle
    """

    print "Reading json file..."
    object = []
    num_not_accepted = 0
    total_processed = 0
    with open(json_file) as json_data:
        for line in json_data:
            if accepted != None:
                element = json.loads(line)
                if accepted(element):
                    object.append(element)
                else:
                    num_not_accepted += 1
                    sys.stdout.write('Not accepted objects: %d / %d \r' % (num_not_accepted, total_processed))
                    sys.stdout.flush()
            else:
                object.append(json.loads(line))
            total_processed += 1

    print "Shuffling resulting python objects"
    random.shuffle(object)

    length_partition = len(object)/num_partitions
    remaining_to_process = len(object)
    current_partition = 1
    while remaining_to_process > 0:
        sys.stdout.write('Importing package %d out of %d \r' % (current_partition, num_partitions))
        sys.stdout.flush()

        # All the remaining elements go to the last partition
        if current_partition == num_partitions:
            stop = None
            num_in_partition = remaining_to_process
        else:
            stop = -remaining_to_process + length_partition
            num_in_partition = length_partition

        pickle.dump(object[-remaining_to_process:stop],
                    open(pickle_name + '.' + str(current_partition), "wb"),
                    pickle.HIGHEST_PROTOCOL)

        current_partition += 1
        remaining_to_process -= num_in_partition

def load_partitions(partition_list, pickle_base_name=DEFAULT_REVIEWS_PICKLE + '.'):
    """
    Returns a python object being a list of dictionaries.
    It reads the data from a sequence of files starting with the given base name. For instance:
    partition_list = [2,4,6], pickle_base_name = "pickle." will read files pickle.2, pickle.4, pickle.6
    """

    num_partition = 1
    result = []
    for partition in partition_list:
        print 'Reading partition %d of %d' % (num_partition, len(partition_list))
        with open(pickle_base_name + str(partition)) as file:
            loaded_element = pickle.load(file)
            result.extend(loaded_element)

        num_partition += 1

    print "Read a total of %d partitions for a total of %d objects" % (num_partition - 1, len(result))
    return result

def accept_only_english(json_review):
    # Short reviews are hard to classify in any language, so they will be accepted
    if len(json_review['text']) <= 150:
        return True
    else:
        return language.detect_language(json_review['text']) == 'english'