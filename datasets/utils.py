import random
from collections import defaultdict

def split_dataset_by_label(data_source):
    """Split a dataset, i.e. a list of Datum objects,
    into class-specific groups stored in a dictionary.

    Args:
        data_source (list): a list of Datum objects.
    """
    output = defaultdict(list)
    container = []

    for item in data_source:
        output[item.label].append(item)
        container.append(item)
    return output, container
    
def generate_noisy_fewshot_dataset(*data_sources, num_shots=-1, num_fp=0, repeat=True):
        """Generate a few-shot dataset with noise (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a few number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            num_fp (int): number of false positive samples per class.
            repeat (bool): repeat images if needed (default: False).
        """
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f"Creating a {num_shots}-shot dataset with {num_fp}-shot noisy label")

        output = []

        for data_source in data_sources:
            tracker, container = split_dataset_by_label(data_source)  # clabel: Datum, Datum
            dataset = []
            sample_chosen_impath = []

            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items

                # produce noisy labels
                noisy_container = [item for item in container if item not in items and item.impath not in sample_chosen_impath]
                noisy_items = random.sample(noisy_container, num_fp)
                for id, item in enumerate(random.sample(sampled_items, num_fp)):
                    item._impath = noisy_items[id].impath
                    sample_chosen_impath.append(noisy_items[id].impath)
                
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]
        
        return output