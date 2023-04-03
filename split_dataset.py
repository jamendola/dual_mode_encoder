import os
import random

if __name__ == "__main__":

    data_path = "../kaist/kaist-cvpr15"
    set_name = "train-all-20.txt"
    address_list = list()
    set_path = os.path.join(data_path, 'imageSets', set_name)
    with open(set_path, 'r') as ft:
        for idx, line in enumerate(ft):
            line = line.strip("\n")
            address_list.append(line)
    size = len(address_list)
    train_cut = set(random.sample(address_list, int(0.8*size)))
    tmp_cut = set(address_list) - train_cut
    size = len(tmp_cut)
    val_cut = set(random.sample(tmp_cut, int(0.5*size)))
    test_cut = tmp_cut - val_cut
    print('val_cut ', len(val_cut), 'test_cut ', len(test_cut), 'train_cut ', len(train_cut))
    with open(os.path.join(data_path, 'train_cut'), 'w') as fp:
        for item in train_cut:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')
    with open(os.path.join(data_path, 'test_cut'), 'w') as fp:
        for item in test_cut:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')
    with open(os.path.join(data_path, 'val_cut'), 'w') as fp:
        for item in val_cut:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')
