import six.moves.cPickle as pickle
import time
import os


def gen_cascade_graph(observation_time, pre_times, filename, filename_ctrain, filename_cval, filename_ctest):
    file = open(filename)
    isExists = os.path.exists(DATA_PATHA)
    if not isExists:
        os.makedirs(DATA_PATHA)
    file_ctrain = open(filename_ctrain, 'w')
    file_cval = open(filename_cval, "w")
    file_ctest = open(filename_ctest, "w")
    cascades_total = dict()
    for line in file:
        parts = line.split("\t")
        if len(parts) != 5:
            print('wrong format!')
            continue
        cascadeID = parts[0]
        # print cascadeID
        n_nodes = int(parts[3])     # retweet number
        path = parts[4].split(" ")
        if n_nodes != len(path):
            print('wrong number of nodes', n_nodes, len(path))
            # continue
        msg_time = int(parts[2])
        hour = time.strftime("%H", time.localtime(msg_time))
        hour = int(hour)
        # print msg_time,hour
        if hour <= 7 or hour >= 19:
            continue
        observation_path = []
        labels = []
        # edges = set()
        for i in range(len(pre_times)):
            labels.append(0)
        for p in path:
            nodes = p.split(":")[0].split("/")
            # print nodes
            time_now = int(p.split(":")[1])     # retweet time
            if time_now < observation_time:
                observation_path.append(",".join(nodes)+":"+str(time_now))
            for i in range(len(pre_times)):
                if time_now < pre_times[i]:
                    labels[i] += 1

        if len(observation_path) < 10 or len(observation_path) > 1000:
            continue
        cascades_total[cascadeID] = msg_time

    n_total = len(cascades_total) 
    print('total:', n_total)
    cascades_type = dict()
    sorted_msg_time = sorted(cascades_total.items(), key=lambda x: x[1])
    count = 0
    for (k, v) in sorted_msg_time:
        if count < n_total*1.0/20*14:
            cascades_type[k] = 1
        elif count < n_total*1.0/20*17:
            cascades_type[k] = 2
        else:
            cascades_type[k] = 3
        count += 1

    file.close()
    file = open(filename, "r")

    zero_lable_cnt = 0
    z_val = 0
    z_train = 0
    z_test = 0
    sum_label_train = 0
    sum_label_test = 0
    sum_label_val = 0
    cnt_label_train = 0
    cnt_label_test = 0
    cnt_label_val = 0
    for line in file:
        parts = line.split("\t")
        if len(parts) != 5:
            print('wrong format!')
            continue
        cascadeID = parts[0]
        n_nodes = int(parts[3])
        path = parts[4].split(" ")
        # fans_num = parts[5].split(" ")
        if n_nodes != len(path):
            print('wrong number of nodes', n_nodes, len(path))
            # continue
        # msg_time = time.localtime(int(parts[2]))
        # print msg_time
        # hour = time.strftime("%H", msg_time)
        observation_path = []
        labels = []
        # edges = set()
        edges = dict()
        for i in range(len(pre_times)):
            labels.append(0)
        for k in range(len(path)):
            nodes = path[k].split(":")[0].split("/")
            time_now = int(path[k].split(":")[1])
            if time_now < observation_time:
                observation_path.append(",".join(nodes)+":"+str(time_now))
                if len(nodes) == 1:
                    edges[nodes[0]+":"+nodes[0]] = 0
                else:
                    if nodes[-2] + ":" + nodes[-1] not in edges:
                        edges[nodes[-2] + ":" + nodes[-1]] = time_now
            for i in range(len(pre_times)):
                # print time,pre_times[i]
                if time_now < pre_times[i]:
                    labels[i] += 1
        edges = dict(sorted(edges.items(), key=lambda x: x[1]))
        temp_cascade = []
        for path_key, path_value in edges.items():
            temp_cascade.append(str(path_key) + ':' + str(path_value))
        for i in range(len(labels)):
            labels[i] = str(labels[i]-len(observation_path))
        if int(labels[-1]) < 1 and cascadeID in cascades_type:
            zero_lable_cnt += 1
            if cascades_type[cascadeID] == 1:
                z_train += 1
                if z_train % 2 != 0:
                    continue
            if cascades_type[cascadeID] == 2:
                z_val += 1
                if z_val % 2 != 0:
                    continue
            if cascades_type[cascadeID] == 3:
                z_test += 1
                if z_test % 2 != 0:
                    continue
            # continue
        if cascadeID in cascades_type and cascades_type[cascadeID] == 1:
            file_ctrain.write(cascadeID+"\t"+parts[1]+"\t"+parts[2]+"\t"+str(len(observation_path))+"\t"
                              + " ".join(temp_cascade) + "\t" + " ".join(labels)+"\n")
            sum_label_train += len(observation_path)
            cnt_label_train += 1
        elif cascadeID in cascades_type and cascades_type[cascadeID] == 2:
            file_cval.write(cascadeID + "\t" + parts[1] + "\t" + parts[2] + "\t" + str(len(observation_path)) + "\t"
                            + " ".join(temp_cascade) + "\t" + " ".join(labels) + "\n")
            sum_label_val += len(observation_path)
            cnt_label_val += 1
        elif cascadeID in cascades_type and cascades_type[cascadeID] == 3:
            file_ctest.write(cascadeID + "\t" + parts[1] + "\t" + parts[2] + "\t" + str(len(observation_path)) + "\t"
                             + " ".join(temp_cascade) + "\t" + " ".join(labels) + "\n")
            sum_label_test += len(observation_path)
            cnt_label_test += 1

    file.close()
    file_ctrain.close()
    file_cval.close()
    file_ctest.close()
    print('zero_lable_cnt', zero_lable_cnt, z_train, z_val, z_test)
    print('train,', sum_label_train, cnt_label_train, sum_label_train * 1.0 / cnt_label_train)
    print('val,', sum_label_val, cnt_label_val, sum_label_val * 1.0 / cnt_label_val)
    print('test,', sum_label_test, cnt_label_test, sum_label_test * 1.0 / cnt_label_test)


if __name__ == "__main__":
    observation_time = 10800     # set the observation time window T
    DATA_PATHA = "../data/weibo_data_" + str(int(observation_time/3600)) + "h_24h/"
    print('yes', DATA_PATHA)
    # pre_times = [temp*3600 for temp in range(3, 25, 3)]
    pre_times = [24*3600]
    filename = '../data/weibo_data/dataset_weibo.txt'
    filename_ctrain = DATA_PATHA + "cascade_train.txt"
    filename_cval = DATA_PATHA + "cascade_val.txt"
    filename_ctest = DATA_PATHA + "cascade_test.txt"
    gen_cascade_graph(observation_time, pre_times, filename, filename_ctrain, filename_cval, filename_ctest)
