import json
import pickle
import random

DEFAULT_REVIEWS_FILE = "data/yelp_academic_dataset_review.json"
DEFAULT_REVIEWS_PICKLE = "data/reviews.pickle"

def pickles_from_json(json_file=DEFAULT_REVIEWS_FILE, pickle_name=DEFAULT_REVIEWS_PICKLE, num_partitions=100):
    """
    Dumps a json into a number of pickle partitions, which contain a list of python objects.
    """

    print "Reading json file..."
    object = []
    with open(json_file) as json_data:
        for line in json_data:
            object.append(json.loads(line))

    print "Shuffling resulting python objects"
    random.shuffle(object)

    length_partition = len(object)/num_partitions
    remaining_to_process = len(object)
    current_partition = 1
    while remaining_to_process > 0:
        print 'Working on partition {} of {}'.format(current_partition, num_partitions)

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

lol = load_partitions(range(1,101))

