import numpy as np

def inspect_one_subject(dataset, subject_id, max_trials=3):
    data_dict, label_dict = dataset.__get_subject__(subject_id)

    print(f"\n[INSPECT TRANSFORM] Subject {subject_id}")
    print("-" * 60)

    for i, (trial_id, data) in enumerate(data_dict.items()):
        # data shape: (n_segments, 1, channels, time)
        n_segments = data.shape[0]
        channels = data.shape[2]
        time_samples = data.shape[3]

        print(f"Trial {trial_id}")
        print(f"  Segments       : {n_segments}")
        print(f"  Channels       : {channels}")
        print(f"  Samples/segment: {time_samples}")
        print(f"  Label          : {label_dict[trial_id]}")
        print("-" * 60)

        if i + 1 >= max_trials:
            break


def count_total_segments(dataset):
    total_segments = 0

    for subject_id in dataset.__get_subject_ids__():
        data_dict, _ = dataset.__get_subject__(subject_id)
        for trial_data in data_dict.values():
            # trial_data shape: (num_segments, 1, channels, time)
            total_segments += trial_data.shape[0]

    return total_segments