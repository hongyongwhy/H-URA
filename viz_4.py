import time, random, logging, argparse
from DataHelper import Dataset, load_embedding, load_user_product_embeddings
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pdb
import statistics
import math
import collections
from opinion_datasets import Opinion_Lexicons
from text_attention import generate
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.it'] = 'Arial:italic'
rcParams['mathtext.rm'] = 'Arial'


def index_to_vocab(w_idx, vocab_keys, vocab_values):
    w_idx_idx = vocab_values.index(w_idx)
    word = vocab_keys[w_idx_idx]
    return word


def save(path, obj):
    pickle.dump(obj, open(path, "wb"))


def load(path):
    return pickle.load(open(path, 'rb'))


def create_data_set(model_idx):
    # 1. load dataset
    # 2. load viz_dataset
    # 3. load opinion dataset
    # 4. store
    data_dir = './acl2015/yelp14/'
    word_embed_path = data_dir + 'yelp-2014-embedding-200d.txt'
    usr_pdr_embed_path = data_dir + 'yelp14_mf.pkl'
    data_path = [data_dir + 'yelp-2014-seg-20-20.train.ss', data_dir + 'yelp-2014-seg-20-20.dev.ss',
                 data_dir + 'yelp-2014-seg-20-20.test.ss']
    # model_idx = 428

    viz_data_path = ['viz_data/yelp14/yelp14_train_viz_data_{}.pkl'.format(model_idx),
                     'viz_data/yelp14/yelp14_dev_viz_data_{}.pkl'.format(model_idx),
                     'viz_data/yelp14/yelp14_test_viz_data_{}.pkl'.format(model_idx)]

    max_doc_len, max_sen_len = 40, 50

    doc_embed, vocab_dict = load_embedding(word_embed_path, 200)
    vocab_dict_key_list, vocab_dict_value_list = list(vocab_dict.keys()), list(vocab_dict.values())
    [usr_dict, pdr_dict], [user_embed, product_embed], [usr_rate_dist, pdr_rate_dist] = load_user_product_embeddings(
        file_path=usr_pdr_embed_path)

    train_set = Dataset(data_path[0], max_doc_len, max_sen_len, vocab_dict, usr_dict, pdr_dict,
                        usr_rate_dist_list=usr_rate_dist, pdr_rate_dist_list=pdr_rate_dist)
    dev_set = Dataset(data_path[1], max_doc_len, max_sen_len, vocab_dict, usr_dict, pdr_dict,
                      usr_rate_dist_list=usr_rate_dist, pdr_rate_dist_list=pdr_rate_dist)
    test_set = Dataset(data_path[2], max_doc_len, max_sen_len, vocab_dict, usr_dict, pdr_dict,
                       usr_rate_dist_list=usr_rate_dist, pdr_rate_dist_list=pdr_rate_dist)

    train_set.genBatch(batch_size=32)
    dev_set.genBatch(batch_size=32)
    test_set.genBatch(batch_size=32)

    train_viz_data = pickle.load(open(viz_data_path[0], 'rb'))
    dev_viz_data = pickle.load(open(viz_data_path[1], 'rb'))
    test_viz_data = pickle.load(open(viz_data_path[2], 'rb'))

    print(train_viz_data.keys())

    lex = Opinion_Lexicons()
    liu_lex_corpus = lex.load_liu_lex()

    # train_viz_data = train_viz_data['data']
    # dev_viz_data = dev_viz_data['data']
    # test_viz_data = test_viz_data['data']

    word_stop_sign = '$EOF$'
    final_output_list = []

    for (raw_data, viz_data) in zip([train_set, dev_set, test_set], [train_viz_data, dev_viz_data, test_viz_data]):
        print(viz_data['name'])
        for b_idx, batched_data in enumerate(raw_data.gen_batched_data):

            _, word_score_list, sen_score_list = viz_data['data'][b_idx]['batch_idx'], viz_data['data'][b_idx][
                'word_score'], viz_data['data'][b_idx]['sen_score']

            tol_sen = 0
            for inst_idx, inst in enumerate(batched_data):
                uid = inst[0]
                pid = inst[1]
                doc = inst[5]
                doc_org = inst[7].split('<sssss>')
                doc_sen_len = inst[6]

                label_0 = inst[4]

                start_sen_idx, end_sen_idx = tol_sen, tol_sen + len(doc_sen_len)
                tol_sen += len(doc_sen_len)
                curr_word_score = word_score_list[start_sen_idx:end_sen_idx, :]
                # print(np.sum(curr_word_score, axis=-1))
                curr_sen_score = sen_score_list[inst_idx][0:len(doc_sen_len)]
                # print(sum(curr_sen_score))

                curr_sen_score_order = np.argsort(-curr_sen_score)
                review_words_actual = []
                review_words_score_actual = []

                doc_proc = []

                for s_idx, s in enumerate(doc):
                    s_score = curr_sen_score[s_idx]
                    s_score_order = curr_sen_score_order[s_idx]

                    s_words_temp = [index_to_vocab(w, vocab_dict_key_list, vocab_dict_value_list) for w in s]
                    end_word_idx = s_words_temp.index(word_stop_sign) if word_stop_sign in s_words_temp else len(
                        s_words_temp)
                    s_words = s_words_temp[0:end_word_idx]

                    # pdb.set_trace()

                    words_scores_temp = np.squeeze(curr_word_score[s_idx])
                    words_score = words_scores_temp[0: len(s_words)]
                    word_score_order = np.argsort(-words_score)
                    # print(sum(words_score))
                    output_words = []
                    # print sentence with importance score
                    for word_idx, word in enumerate(s_words):
                        word_s = words_score[word_idx]
                        word_order = np.where(word_score_order == word_idx)[0].tolist()[0]
                        word_score = words_score[word_idx]

                        word_s_output = '{} ({})'.format(word, word_order)
                        output_words.append(word_s_output)

                    output_sentence = '(#{}) {}'.format(s_score_order, ' '.join(output_words))
                    doc_proc.append(output_sentence)

                    # for _ in doc_org_data:
                    #     logging.info(_.strip())
                    # label_1 = labels[review_instance_idx]
                    # pred_label = pred_labels[review_instance_idx]

                    # doc_sen_len = review_instance[6]
                    # start_sen_idx, end_sen_idx = tol_sen, tol_sen + len(doc_sen_len)
                    # tol_sen += len(doc_sen_len)
                    # curr_word_score = word_score_list[start_sen_idx:end_sen_idx, :]
                    # # print(np.sum(curr_word_score, axis=-1))
                    # curr_sen_score = sen_score_list[review_instance_idx][0:len(doc_sen_len)]
                    # curr_sen_score_order = np.argsort(-curr_sen_score)

                    # print(np.sum(curr_sen_score, axis=-1))
                    # print()

                    # print(sum(curr_sen_score) -  1)
                    # if user_id != 10:
                    #     continue

                # find neg and pos
                pos_terms = []
                neg_terms = []
                for sen in doc_org:
                    for word in sen.split():
                        polarity = liu_lex_corpus.get(word, None)
                        if polarity is not None:
                            if polarity == 'pos':
                                pos_terms.append(word)
                            elif polarity == 'neg':
                                neg_terms.append(word)

                final_output = {'pid': pid, 'uid': uid, 'doc_org': doc_org, 'doc_proc': doc_proc, 'neg_terms': neg_terms,
                                'pos_terms': pos_terms, 'rating': label_0}
                final_output_list.append(final_output)

    print(len(final_output_list))
    print('finished')

    save(path='viz_log/yelp14_all_{}.pkl'.format(model_idx), obj=final_output_list)
    obj = load(path='viz_log/yelp14_all_{}.pkl'.format(model_idx))
    print(len(obj))

def find_all_pairs():
    obj_list = load('viz_log/yelp14_all_428.pkl')
    print(len(obj_list))
    all_pairs = []
    # by pairs {a: , b:}
    for idx, inst1 in enumerate(obj_list):
        if idx % 10000 == 0: print(idx)
        # if idx > 10: break
        pid1 = inst1['pid']
        uid1 = inst1['uid']
        # doc_org = inst1['doc_org']
        neg_terms1 = inst1['neg_terms']
        pos_terms1 = inst1['pos_terms']
        for inst2 in obj_list:
            pid2 = inst2['pid']
            uid2 = inst2['uid']
            # doc_org = inst2['doc_org']
            neg_terms2 = inst2['neg_terms']
            pos_terms2 = inst2['pos_terms']

            if pid1 == pid2 and uid1 != uid2 and len(neg_terms1) == len(neg_terms2) and len(pos_terms1) == len(
                    pos_terms2):
                out_inst = {'u1': uid1, 'u2': uid2, 'pid': pid1, 'u1_inst': inst1, 'u2_inst': inst2}
                all_pairs.append(out_inst)

    save('viz_log/yelp14_all_user_diff_pairs.pkl', all_pairs)
    print('finished')

def plot_1_words():
    all_pairs = load('viz_log/yelp14_all_user_diff_pairs.pkl')
    for idx, pair in enumerate(all_pairs):
        u1 = pair['u1']
        u2 = pair['u2']
        pid = pair['pid']
        u1_inst = pair['u1_inst']
        u2_inst = pair['u2_inst']

        # find overlapping of sentiment words
        u1_inst_pos = u1_inst['pos_terms']
        u2_inst_pos = u2_inst['pos_terms']

        common_found = []
        for term in u1_inst_pos:
            if term in u2_inst_pos:
                common_found.append(term)

        if len(common_found) == 0: continue

        p1 = len(common_found) / len(u1_inst_pos)
        p2 = len(common_found) / len(u2_inst_pos)

        distinct_common_terms = collections.Counter(common_found).keys()

        if p1 > 0.8 and p2 > 0.8 and len(distinct_common_terms) > 4 \
                and abs(u1_inst['rating'] - u2_inst['rating']) >= 1:
            print(u1_inst)
            print(u2_inst)
            print(distinct_common_terms)
            print('*' * 80)

    print('finished')


def yelp14_all_sample_terms(model_idx):

    # model_idx = 327 # 428
    yelp14_all_data = load('viz_log/yelp14_all_{}.pkl'.format(model_idx))
    # find user keys
    pos_terms_by_user = dict()
    neg_terms_by_user = dict()

    for inst in yelp14_all_data:
        uid = inst['uid']
        pid = inst['pid']
        doc_org = inst['doc_org']
        rating = inst['rating']

        all_pos_terms, all_neg_terms = [], []
        all_neg_terms.extend(inst['neg_terms'])
        all_pos_terms.extend(inst['pos_terms'])

        if pos_terms_by_user.get(uid, None) is None:
            pos_terms_by_user[uid] = []

        if neg_terms_by_user.get(uid, None) is None:
            neg_terms_by_user[uid] = []

        pos_terms_by_user[uid].extend(all_pos_terms)
        neg_terms_by_user[uid].extend(all_neg_terms)

    # for k, v in pos_terms_by_user.items():
    #     pos_terms = list(collections.Counter(v).keys())
    #     if len(pos_terms) < 50 and len(v) > 200: # [568, 901, 1289, 1973, 2177, 2822, 3375, 3760, 4070, 4137, 4179, 4506, 4723, 4726]
    #         print(k)

    pos_keys = [568, 901, 1289, 1973, 2177, 2822, 3375, 3760, 4070, 4137, 4179, 4506, 4723, 4726] # len(pos_terms) < 50 and len(v) > 200
    all_pos_terms = []
    all_neg_terms = []
    for key in pos_keys:

        rating_pos = dict()

        for inst in yelp14_all_data:
            uid = inst['uid']
            pid = inst['pid']
            doc_org = inst['doc_org']
            rating = inst['rating']
            if uid == key:
                if rating_pos.get(rating, None) is None:
                    rating_pos[rating] = []

                rating_pos[rating].extend(inst['pos_terms'])
                all_pos_terms.extend(inst['pos_terms'])
                all_neg_terms.extend(inst['neg_terms'])

        # for k, v in rating_pos.items():
        #     print('{}: {}'.format(k, collections.Counter(v)))

    yelp14_all_sample_terms = {'pos_keys': pos_keys, 'all_pos_terms': all_pos_terms, 'all_neg_terms': all_neg_terms}
    save('viz_log/yelp14_all_{}_sample_terms.pkl'.format(model_idx), yelp14_all_sample_terms)


def find_yelp14_all_sample_terms_data(model_idx):
    yelp14_all_sample_terms = load('viz_log/yelp14_all_{}_sample_terms.pkl'.format(model_idx))
    # pos_keys = [568, 901, 1289, 1973, 2177, 2822, 3375, 3760, 4070, 4137, 4179, 4506, 4723, 4726]
    pos_keys = yelp14_all_sample_terms['pos_keys']
    # print(collections.Counter(yelp14_all_sample_terms['all_pos_terms']))
    # pos_terms = list(collections.Counter(yelp14_all_sample_terms['all_pos_terms']).keys())
    # neg_terms = list(collections.Counter(yelp14_all_sample_terms['all_neg_terms']).keys())

    # ---
    data_dir = './acl2015/yelp14/'
    word_embed_path = data_dir + 'yelp-2014-embedding-200d.txt'
    usr_pdr_embed_path = data_dir + 'yelp14_mf.pkl'
    data_path = [data_dir + 'yelp-2014-seg-20-20.train.ss', data_dir + 'yelp-2014-seg-20-20.dev.ss',
                 data_dir + 'yelp-2014-seg-20-20.test.ss']

    viz_data_path = ['viz_data/yelp14/yelp14_train_viz_data_{}.pkl'.format(model_idx),
                     'viz_data/yelp14/yelp14_dev_viz_data_{}.pkl'.format(model_idx),
                     'viz_data/yelp14/yelp14_test_viz_data_{}.pkl'.format(model_idx)]

    max_doc_len, max_sen_len = 40, 50

    doc_embed, vocab_dict = load_embedding(word_embed_path, 200)
    vocab_dict_key_list, vocab_dict_value_list = list(vocab_dict.keys()), list(vocab_dict.values())
    [usr_dict, pdr_dict], [user_embed, product_embed], [usr_rate_dist, pdr_rate_dist] = load_user_product_embeddings(
        file_path=usr_pdr_embed_path)

    train_set = Dataset(data_path[0], max_doc_len, max_sen_len, vocab_dict, usr_dict, pdr_dict,
                        usr_rate_dist_list=usr_rate_dist, pdr_rate_dist_list=pdr_rate_dist)
    dev_set = Dataset(data_path[1], max_doc_len, max_sen_len, vocab_dict, usr_dict, pdr_dict,
                      usr_rate_dist_list=usr_rate_dist, pdr_rate_dist_list=pdr_rate_dist)
    test_set = Dataset(data_path[2], max_doc_len, max_sen_len, vocab_dict, usr_dict, pdr_dict,
                       usr_rate_dist_list=usr_rate_dist, pdr_rate_dist_list=pdr_rate_dist)

    train_set.genBatch(batch_size=32)
    dev_set.genBatch(batch_size=32)
    test_set.genBatch(batch_size=32)

    train_viz_data = pickle.load(open(viz_data_path[0], 'rb'))
    dev_viz_data = pickle.load(open(viz_data_path[1], 'rb'))
    test_viz_data = pickle.load(open(viz_data_path[2], 'rb'))

    print(train_viz_data.keys())

    lex = Opinion_Lexicons()
    liu_lex_corpus = lex.load_liu_lex()

    word_stop_sign = '$EOF$'
    final_output_list = []

    for (raw_data, viz_data) in zip([train_set, dev_set, test_set], [train_viz_data, dev_viz_data, test_viz_data]):
        print(viz_data['name'])
        for b_idx, batched_data in enumerate(raw_data.gen_batched_data):

            _, word_score_list, sen_score_list = viz_data['data'][b_idx]['batch_idx'], viz_data['data'][b_idx][
                'word_score'], viz_data['data'][b_idx]['sen_score']

            tol_sen = 0
            for inst_idx, inst in enumerate(batched_data):
                uid = inst[0]

                if uid in pos_keys:
                    pid = inst[1]
                    doc = inst[5]
                    doc_org = inst[7].split('<sssss>')
                    doc_sen_len = inst[6]

                    label_0 = inst[4]

                    start_sen_idx, end_sen_idx = tol_sen, tol_sen + len(doc_sen_len)
                    tol_sen += len(doc_sen_len)
                    curr_word_score = word_score_list[start_sen_idx:end_sen_idx, :]
                    # print(np.sum(curr_word_score, axis=-1))
                    curr_sen_score = sen_score_list[inst_idx][0:len(doc_sen_len)]
                    # print(sum(curr_sen_score))

                    curr_sen_score_order = np.argsort(-curr_sen_score)
                    review_words_actual = []
                    review_words_score_actual = []

                    doc_proc = []

                    for s_idx, s in enumerate(doc):
                        s_score = curr_sen_score[s_idx]
                        s_score_order = curr_sen_score_order[s_idx]

                        s_words_temp = [index_to_vocab(w, vocab_dict_key_list, vocab_dict_value_list) for w in s]
                        end_word_idx = s_words_temp.index(word_stop_sign) if word_stop_sign in s_words_temp else len(
                            s_words_temp)
                        s_words = s_words_temp[0:end_word_idx]

                        # pdb.set_trace()

                        words_scores_temp = np.squeeze(curr_word_score[s_idx])
                        words_score = words_scores_temp[0: len(s_words)]
                        word_score_order = np.argsort(-words_score)
                        # print(sum(words_score))
                        output_words = []
                        # print sentence with importance score
                        for word_idx, word in enumerate(s_words):
                            word_s = words_score[word_idx]
                            word_order = np.where(word_score_order == word_idx)[0].tolist()[0]
                            word_score = words_score[word_idx]
                            output_words.append([word, word_order, word_score])

                        output_sentence = [output_words, s_score_order, s_score]  # '(#{}) {}'.format(s_score_order, ' '.join(output_words))
                        doc_proc.append(output_sentence)

                    output_inst = {'uid': uid, 'pid': pid, 'doc': doc_proc, 'rating': label_0}
                    final_output_list.append(output_inst)

        save('viz_log/yelp14_all_{}_sample_terms_doc_data.pkl'.format(model_idx), final_output_list)
        final_output_list = load('viz_log/yelp14_all_{}_sample_terms_doc_data.pkl'.format(model_idx))
        # print('finished')

def plot_sentiment_words_comparision_0(model_idx):
    # create_data_set()
    # yelp14_all_428.pkl
    # yelp14_all_user_diff_pairs.pkl
    # store all dataset in format: pid, uid, org_doc, doc_proc, neg, pos
    yelp14_all = load('viz_log/yelp14_all_{}.pkl'.format(model_idx))

    yelp14_all_sample_terms = load('viz_log/yelp14_all_{}_sample_terms.pkl'.format(model_idx))
    # pos_keys = [568, 901, 1289, 1973, 2177, 2822, 3375, 3760, 4070, 4137, 4179, 4506, 4723, 4726]
    pos_keys = yelp14_all_sample_terms['pos_keys']
    # print(collections.Counter(yelp14_all_sample_terms['all_pos_terms']))
    pos_terms = list(collections.Counter(yelp14_all_sample_terms['all_pos_terms']).keys())
    neg_terms = list(collections.Counter(yelp14_all_sample_terms['all_neg_terms']).keys())

    # opinion_corpus = pos_terms
    opinion_corpus = list(set(pos_terms + neg_terms))
    # opinion_corpus = opinion_corpus[0:100]
    print(len(pos_terms))

    for idx, key in enumerate(pos_keys):
        # if idx >= 1 : break
        # if key != 3760: continue
        if key not in [4506, 3375]: continue
        tol_reviews = 0
        inst_plot_data = dict()
        inst_plot_data_filted = dict()

        for inst in yelp14_all:

            uid = inst['uid']
            pid = inst['pid']
            doc_org = inst['doc_org']
            rating = inst['rating']
            if uid == key and rating in [0, 1, 2, 3, 4]:
                tol_reviews += 1
                inst_pos_terms = inst['pos_terms']
                inst_neg_terms = inst['neg_terms']
                opinion_terms = inst_pos_terms + inst_neg_terms
                for term in opinion_terms:
                    if term in opinion_corpus and term not in ['hot', 'right', 'split', 'funky']:
                        if inst_plot_data.get(term, None) is None:
                            inst_plot_data[term] = [0, 0, 0, 0, 0]

                        inst_plot_data[term][rating] += 1

        for w, dist in inst_plot_data.items():
            cnt = sum([1 if c > 0 else 0 for c in dist])
            if cnt > 2 and key == 4506:
                inst_plot_data_filted[w] = [v if v > 0 else 0 for v in dist]
            elif cnt > 1 and key == 3375:
                inst_plot_data_filted[w] = [v if v > 0 else 0 for v in dist]

        #sort by keys
        inst_plot_data_filted_sorted = sorted(inst_plot_data_filted.items(), reverse=False)
        words = [item[0] for item in inst_plot_data_filted_sorted]
        # values = list(inst_plot_data.values())
        print(len(words))
        print(inst_plot_data_filted)
        if len(words) == 0:
            print('no words')
            continue

        r1 = [item[1][0] for item in inst_plot_data_filted_sorted]
        r2 = [item[1][1] for item in inst_plot_data_filted_sorted]
        r3 = [item[1][2] for item in inst_plot_data_filted_sorted]
        r4 = [item[1][3] for item in inst_plot_data_filted_sorted]
        r5 = [item[1][4] for item in inst_plot_data_filted_sorted]

        # zip(r1, r2, r3, r4, r5)
        # group_words = [r1, r2, r3, r4, r5]
        # num_groups = len(group_words)
        # ind = range(num_groups)
        # for g_idx, vals in enumerate(group_words):
        #     group_len = len(vals)
        #     plt.bar(x=ind, height=vals, width=barWidth, align='center')
        #     ind = [i + barWidth for i in ind]
        #
        # offset = (num_groups / 2.) * barWidth
        # xticks = [x + offset for x in range(num_groups)]
        # plt.xticks(xticks)
        # plt.xlim([0 - barWidth, max(xticks) + (num_groups * barWidth)])
        # plt.xticklabels(words)
        #
        #
        #
        if key == 4506:
            barWidth = 0.8
            xp1 = np.arange(0, 5 * len(r1), 5)
            plt.xticks([r - 0.1 + barWidth * 5 / 2 for r in xp1], words, rotation=85)
        elif key == 3375:
            barWidth = 0.9
            xp1 = np.arange(0, 6 * len(r1), 6)
            plt.xticks([r - 0.15 + barWidth * 6 / 2 for r in xp1], words, rotation=85)

        xp2 = [x + barWidth for x in xp1]
        # xp3 = np.arange(len(r3))
        xp3 = [x + barWidth for x in xp2]
        xp4 = [x + barWidth for x in xp3]
        xp5 = [x + barWidth for x in xp4]

        plt.bar(xp1, r1, color='#C1C1C4', width=barWidth, edgecolor='white', label='rating 1')
        plt.bar(xp2, r2, color='#9BFFE8', width=barWidth, edgecolor='white', label='rating 2')
        plt.bar(xp3, r3, color='#5EC7C0', width=barWidth, edgecolor='white', label='rating 3')
        plt.bar(xp4, r4, color='#44938E', width=barWidth, edgecolor='white', label='rating 4')
        plt.bar(xp5, r5, color='#275755', width=barWidth, edgecolor='white', label='rating 5')
        # plt.xlim(-0.5, len(r1) + len(r2) + len(r3) + len(r4) + len(r5) + 5.0)
        if key == 4405:
            plt.yticks(np.arange(0, 20, step=2))
        elif key == 3375:
            plt.yticks(np.arange(0, 40, step=5))

        plt.tick_params(labelsize=11, axis='x', color='red')
        # plt.xlabel('user #{}'.format(key), fontweight='light', fontsize=14)
        plt.ylabel('Frequency', fontsize=11, position=(1, 0.5))
        plt.legend()
        # # plot_data = pd.DataFrame.from_dict({'word': list(inst_plot_data.keys()), 'count': list(inst_plot_data.values())})
        # # ax = sns .barplot(x=plot_data['word'], y=plot_data['count'], data=plot_data)
        # # ax.set_xticklabels(list(plot_data['word']), rotation=70)
        plt.tight_layout()
        plt.savefig('plot/{}.eps'.format(key), format='eps', dpi=1000)
        print('key:{}, tol_reviews:{}'.format(key, tol_reviews))
        plt.show()


def plot_sentiment_words_comparision(model_idx):
    # create_data_set()
    # yelp14_all_428.pkl
    # yelp14_all_user_diff_pairs.pkl
    # store all dataset in format: pid, uid, org_doc, doc_proc, neg, pos
    yelp14_all = load('viz_log/yelp14_all_{}.pkl'.format(model_idx))

    yelp14_all_sample_terms = load('viz_log/yelp14_all_{}_sample_terms.pkl'.format(model_idx))
    # pos_keys = [568, 901, 1289, 1973, 2177, 2822, 3375, 3760, 4070, 4137, 4179, 4506, 4723, 4726]
    pos_keys = yelp14_all_sample_terms['pos_keys']
    # print(collections.Counter(yelp14_all_sample_terms['all_pos_terms']))
    pos_terms = list(collections.Counter(yelp14_all_sample_terms['all_pos_terms']).keys())
    neg_terms = list(collections.Counter(yelp14_all_sample_terms['all_neg_terms']).keys())

    # opinion_corpus = pos_terms
    opinion_corpus = list(set(pos_terms + neg_terms))
    # opinion_corpus = opinion_corpus[0:100]
    print(len(pos_terms))

    for idx, key in enumerate(pos_keys):
        # if idx >= 1 : break
        # if key != 3760: continue
        if key not in [3375, 4506]: continue
        tol_reviews = 0
        inst_plot_data = dict()
        inst_plot_data_filted = dict()

        for inst in yelp14_all:

            uid = inst['uid']
            pid = inst['pid']
            doc_org = inst['doc_org']
            rating = inst['rating']
            if uid == key and rating in [0, 1, 2, 3, 4]:
                tol_reviews += 1
                inst_pos_terms = inst['pos_terms']
                inst_neg_terms = inst['neg_terms']
                opinion_terms = inst_pos_terms + inst_neg_terms
                for term in opinion_terms:
                    if term in opinion_corpus and term not in ['hot', 'right', 'split', 'funky']:
                        if inst_plot_data.get(term, None) is None:
                            inst_plot_data[term] = [0, 0, 0, 0, 0]

                        inst_plot_data[term][rating] += 1

        for w, dist in inst_plot_data.items():
            cnt = sum([1 if c > 0 else 0 for c in dist])
            if cnt > 2 and key == 4506:
                inst_plot_data_filted[w] = [v if v > 0 else 0 for v in dist]
            elif cnt > 1 and key == 3375:
                inst_plot_data_filted[w] = [v if v > 0 else 0 for v in dist]
        #sort by keys
        inst_plot_data_filted_sorted = sorted(inst_plot_data_filted.items(), reverse=False)
        words = [item[0] for item in inst_plot_data_filted_sorted]
        # values = list(inst_plot_data.values())
        print(len(words))
        print(inst_plot_data_filted)
        if len(words) == 0:
            print('no words')
            continue

        # r1 = [0 if item[1][0] == 0 else math.log(item[1][0], 2) for item in inst_plot_data_filted_sorted]
        # r2 = [0 if item[1][1] == 0 else math.log(item[1][1], 2) for item in inst_plot_data_filted_sorted]
        # r3 = [0 if item[1][2] == 0 else math.log(item[1][2], 2) for item in inst_plot_data_filted_sorted]
        # r4 = [0 if item[1][3] == 0 else math.log(item[1][3], 2) for item in inst_plot_data_filted_sorted]
        # r5 = [0 if item[1][4] == 0 else math.log(item[1][4], 2) for item in inst_plot_data_filted_sorted]

        r1 = [item[1][0] for item in inst_plot_data_filted_sorted]
        r2 = [item[1][1] for item in inst_plot_data_filted_sorted]
        r3 = [item[1][2] for item in inst_plot_data_filted_sorted]
        r4 = [item[1][3] for item in inst_plot_data_filted_sorted]
        r5 = [item[1][4] for item in inst_plot_data_filted_sorted]

        # print(r1)
        # print(r2)
        # print(r3)
        # print(r4)
        # print(r5)
        # zip(r1, r2, r3, r4, r5)
        # group_words = [r1, r2, r3, r4, r5]
        # num_groups = len(group_words)
        # ind = range(num_groups)
        # for g_idx, vals in enumerate(group_words):
        #     group_len = len(vals)
        #     plt.bar(x=ind, height=vals, width=barWidth, align='center')
        #     ind = [i + barWidth for i in ind]
        #
        # offset = (num_groups / 2.) * barWidth
        # xticks = [x + offset for x in range(num_groups)]
        # plt.xticks(xticks)
        # plt.xlim([0 - barWidth, max(xticks) + (num_groups * barWidth)])
        # plt.xticklabels(words)
        #
        # pre-allocation starting position for rating 1 of each instances.
        # the starting position calculated by the number of non-zero entries of the previous instance
        # 1) calculating the number of non-zero entries of each instance
        word_starting_pos = []
        x_word_pos_list = []
        barWidth = 0.15
        space = 0.2
        word_ratings = [item[1] for item in inst_plot_data_filted_sorted]
        for word_idx, _ in enumerate(word_ratings):
            if word_idx == 0:
                std_pos = 0.1
            else:
                vs = [1 if r > 0 else 0 for r in word_ratings[word_idx-1]]
                prev_pos = word_starting_pos[word_idx-1]
                std_pos = prev_pos + sum(vs) * barWidth + space
            word_starting_pos.append(round(std_pos, 2))

        # pdb.set_trace()
        # 2) calculating each rating position
        xp1, xp2, xp3, xp4, xp5 = [], [], [], [], []

        for word_idx, std_pos in enumerate(word_starting_pos):
            vs = word_ratings[word_idx]
            c_1, c_2, c_3, c_4, c_5 = vs[0], vs[1], vs[2], vs[3], vs[4]
            v_xp1, v_xp2, v_xp3, v_xp4, v_xp5 = 0, 0, 0, 0, 0
            if c_1 > 0:
                v_xp1 = std_pos
                xp1.append(v_xp1)
            else:
                xp1.append(0)

            if c_2 > 0:
                v_xp2 = std_pos if v_xp1 == 0 else (v_xp1 + barWidth)
                xp2.append(v_xp2)
            else:
                xp2.append(0)

            if c_3 > 0:
                if v_xp2 == 0 and v_xp1 == 0:
                    v_xp3 = std_pos
                elif v_xp2 > 0:
                    v_xp3 = v_xp2 + barWidth
                elif v_xp1 > 0:
                    v_xp3 = v_xp1 + barWidth
                xp3.append(v_xp3)
            else:
                xp3.append(0)

            if c_4 > 0:
                if v_xp1 == 0 and v_xp2 == 0 and v_xp3 == 0:
                    v_xp4 = std_pos
                elif v_xp3 > 0:
                    v_xp4 = v_xp3 + barWidth
                elif v_xp2 > 0:
                    v_xp4 = v_xp2 + barWidth
                elif v_xp1 > 0:
                    v_xp4 = v_xp1 + barWidth

                xp4.append(v_xp4)
            else:
                xp4.append(0)

            if c_5 > 0:
                if v_xp1 == 0 and v_xp2 == 0 and v_xp3 == 0 and v_xp4 == 0:
                    v_xp5 = std_pos
                elif v_xp4 > 0:
                    v_xp5 = v_xp4 + barWidth
                elif v_xp3 > 0:
                    v_xp5 = v_xp3 + barWidth
                elif v_xp2 > 0:
                    v_xp5 = v_xp2 + barWidth
                elif v_xp1 > 0:
                    v_xp5 = v_xp1 + barWidth
                xp5.append(v_xp5)
            else:
                xp5.append(0)

        # xp1 = [v for v in xp1 if v > 0]
        # xp2 = [v for v in xp2 if v > 0]
        # xp3 = [v for v in xp3 if v > 0]
        # xp4 = [v for v in xp4 if v > 0]
        # xp5 = [v for v in xp5 if v > 0]
        #
        # r1 = [v for v in r1 if v > 0]
        # r2 = [v for v in r2 if v > 0]
        # r3 = [v for v in r3 if v > 0]
        # r4 = [v for v in r4 if v > 0]
        # r5 = [v for v in r5 if v > 0]

        # pdb.set_trace()
        # print(xp1)
        # print(xp2)
        # print(xp3)
        # print(xp4)
        # print(xp5)
        # print('--')
        # print(word_starting_pos)

        for word_idx, (v_xp1, v_xp2, v_xp3, v_xp4, v_xp5) in enumerate(zip(xp1, xp2, xp3, xp4, xp5)):
            if sum(1 if v > 0 else 0 for v in [v_xp1, v_xp2, v_xp3, v_xp4, v_xp5]) == 1:
                x_word_pos = word_starting_pos[word_idx]
            else:
                v_max = max([v_xp1, v_xp2, v_xp3, v_xp4, v_xp5])
                x_word_pos = word_starting_pos[word_idx] + (v_max - word_starting_pos[word_idx]) * 0.5

            x_word_pos_list.append(x_word_pos)

        plt.figure(num=1, figsize=(8, 4))
        plt.tick_params(labelsize=13, axis='x', color='red')
        plt.xticks(x_word_pos_list, words, rotation=85)
        plt.xlim(-word_starting_pos[0], word_starting_pos[-1] + barWidth * 6)
        plt.ylabel('Frequency', fontsize=11, position=(1, 0.5))
        plt.tight_layout()

        plt.bar(xp1, r1, color='#C1C1C4', width=barWidth, edgecolor='white', label='rating 1')
        plt.bar(xp2, r2, color='#9BFFE8', width=barWidth, edgecolor='white', label='rating 2')
        plt.bar(xp3, r3, color='#5EC7C0', width=barWidth, edgecolor='white', label='rating 3')
        plt.bar(xp4, r4, color='#44938E', width=barWidth, edgecolor='white', label='rating 4')
        plt.bar(xp5, r5, color='#275755', width=barWidth, edgecolor='white', label='rating 5')

        plt.legend(loc='upper left', ncol=2)
        plt.yscale('log', basey=2)

        # if key == 4506:
        #     plt.yticks(np.arange(0, 18, step=2))
        # elif key == 3375:
        #     plt.yticks(np.arange(0, 40, step=4))
        # plt.xlabel('user #{}'.format(key), fontweight='light', fontsize=14)
        plt.savefig('plot/{}.eps'.format(key), format='eps', dpi=1000)
        print('key:{}, tol_reviews:{}'.format(key, tol_reviews))
        plt.show()


def create_yelp14_all_sample_terms_doc_data_r():
    yelp14_all_428 = load('viz_log/yelp14_all_428.pkl')
    yelp14_all_sample_terms = load('viz_log/yelp14_all_sample_terms.pkl')
    # pos_keys = [568, 901, 1289, 1973, 2177, 2822, 3375, 3760, 4070, 4137, 4179, 4506, 4723, 4726]
    pos_keys = yelp14_all_sample_terms['pos_keys']
    # # print(collections.Counter(yelp14_all_sample_terms['all_pos_terms']))
    # # pos_terms = list(collections.Counter(yelp14_all_sample_terms['all_pos_terms']).keys())
    # # neg_terms = list(collections.Counter(yelp14_all_sample_terms['all_neg_terms']).keys())
    final_output_list = load('viz_log/yelp14_all_sample_terms_doc_data.pkl')
    # print(len(final_output_list))
    for key in pos_keys:
        with open('viz_log/yelp14_all_sample_terms_doc_data_r_234_u_{}.txt'.format(key), 'w') as writer:
            lines = []
            for inst in final_output_list:
                if inst['uid'] == key:
                    pid = inst['pid']
                    doc = inst['doc']
                    rating = inst['rating']
                    if rating < 2: continue
                    print('uid:{}, pid:{}'.format(key, pid))
                    lines.append(['uid:{}, pid:{}, rating:{}'.format(key, pid, rating)])

                    for [sentence, sentence_order, sentence_score] in doc:
                        # top 5 words, ignoring punctuation
                        if sentence_order > 5: continue

                        punctuation = [',', '.', '!', 'and', 'or', 'the']
                        sorted_words = sorted(sentence, key=lambda x: x[1])
                        sorted_words_ignore_punctuation = [w for w in sorted_words if w[0] not in punctuation][0:5]
                        top_5_words = ['{} (#{})'.format(word, word_order) for [word, word_order, word_score] in
                                       sorted_words_ignore_punctuation]
                        # for [word, word_order, word_score] in sentence:

                        print(' '.join(top_5_words))
                        lines.append(' '.join(top_5_words))
                    print('-' * 10)
                    lines.append('-' * 10)
            print('*' * 100)
            lines.append('*' * 100)
            for line in lines:
                writer.writelines(line)
                writer.writelines('\n')
    print('finished')


def heatmap(s_words, words_score):
    df = pd.DataFrame(data=words_score, index=s_words, columns=['weights'])
    ax = sns.heatmap(df, cmap="YlGnBu", xticklabels=False, yticklabels=True)
    # ax.set_title('{}/{}'.format(s_score_order, len(curr_sen_score_order)))
    plt.show()


def rescale(input_list):
    the_array = np.asarray(input_list)
    the_max = np.max(the_array)
    the_min = np.min(the_array)
    rescale = (the_array - the_min)/(the_max-the_min)*100
    return rescale.tolist()


def generate_tex():
    model_idx = 428

    yelp14_all = load('viz_log/yelp14_all_{}.pkl'.format(model_idx))
    yelp14_all_sample_terms = load('viz_log/yelp14_all_sample_terms.pkl')
    # pos_keys = [568, 901, 1289, 1973, 2177, 2822, 3375, 3760, 4070, 4137, 4179, 4506, 4723, 4726]
    pos_keys = yelp14_all_sample_terms['pos_keys']
    # # print(collections.Counter(yelp14_all_sample_terms['all_pos_terms']))
    # # pos_terms = list(collections.Counter(yelp14_all_sample_terms['all_pos_terms']).keys())
    # # neg_terms = list(collections.Counter(yelp14_all_sample_terms['all_neg_terms']).keys())
    final_output_list = load('viz_log/yelp14_all_{}_sample_terms_doc_data.pkl'.format(model_idx))
    # print(len(final_output_list))
    for key_idx, key in enumerate(pos_keys):
        # if key_idx > 1: break
        # if key != 568: continue
        with open('viz_log/yelp14_all_sample_terms_doc_data_r_234_u_{}.txt'.format(key), 'w') as writer:
            lines = []
            for inst in final_output_list:
                if inst['uid'] == key:
                    pid = inst['pid']
                    doc = inst['doc']
                    rating = inst['rating']
                    if rating in [0, 1, 2, 3, 4]:
                        print('uid:{}, pid:{}'.format(key, pid))
                        lines.append(['uid:{}, pid:{}, rating:{}'.format(key, pid, rating)])
                        lst_words, lst_att_scores = [], []
                        for [sentence, sentence_order, sentence_score] in doc:
                            # top 5 words, ignoring punctuation
                            # if sentence_order > 5: continue
                            words = [word for [word, word_order, word_score] in sentence]
                            scores = [word_score * sentence_score for [word, word_order, word_score] in sentence]

                            lst_words.extend(words)
                            lst_att_scores.extend(scores)

                            # heatmap(words, scores)

                            # punctuation = [',', '.', '!', 'and', 'or', 'the']
                            # sorted_words = sorted(sentence, key=lambda x: x[1])
                            # sorted_words_ignore_punctuation = [w for w in sorted_words if w[0] not in punctuation][0:5]
                            # top_5_words = ['{} (#{})'.format(word, word_order) for [word, word_order, word_score] in sorted_words_ignore_punctuation]
                            # # for [word, word_order, word_score] in sentence:
                            #
                            # print(' '.join(top_5_words))
                            # lines.append(' '.join(top_5_words))

                        list_word_attention_tuple = [(lst_words, lst_att_scores)]
                        generate(list_word_attention_tuple,
                                 'plot/tex/{}/{}/u{}_p{}_r{}.tex'.format(model_idx, key, key, pid, rating))
                        # print('-' * 10)
                        lines.append('-' * 10)
            # print('*' * 100)
            # lines.append('*' * 100)
            # for line in lines:
            #     writer.writelines(line)
            #     writer.writelines('\n')
    print('finished')


if __name__ == '__main__':
    # create_data_set(model_idx=168)
    # yelp14_all_sample_terms(model_idx=327)
    # yelp14_all_sample_terms(model_idx=428)
    # plot_sentiment_words_comparision(model_idx=327)
    # for model_idx in [168, 327, 428]:
    #     final_output_list = load('viz_log/yelp14_all_{}_sample_terms_doc_data.pkl'.format(model_idx))
    #     print(len(final_output_list))
    # print('finished')
    # plot_sentiment_words_comparision(model_idx=428)
    test = load('viz_log/yelp14_all_user_diff_pairs.pkl')
    for inst in test:
        u1_inst = inst['u1_inst']
        u2_inst = inst['u2_inst']
        u1_inst_doc = u1_inst['doc_org']
        u2_inst_doc = u2_inst['doc_org']
        for doc in u1_inst_doc:
            # if 'almost always great ribs' in doc:
            if 'love the different dips' in doc:
                print()


    print()




if __name__ == '__main__2':
    # plot_sentiment_words_comparision()

    # find_yelp14_all_sample_terms_data()
    # create_data_set()
    # yelp14_all_428.pkl
    # yelp14_all_user_diff_pairs.pkl
    # store all dataset in format: pid, uid, org_doc, doc_proc, neg, pos
    model_idx = 428

    yelp14_all = load('viz_log/yelp14_all_{}.pkl'.format(model_idx))
    yelp14_all_sample_terms = load('viz_log/yelp14_all_sample_terms.pkl')
    # pos_keys = [568, 901, 1289, 1973, 2177, 2822, 3375, 3760, 4070, 4137, 4179, 4506, 4723, 4726]
    pos_keys = yelp14_all_sample_terms['pos_keys']
    # # print(collections.Counter(yelp14_all_sample_terms['all_pos_terms']))
    # # pos_terms = list(collections.Counter(yelp14_all_sample_terms['all_pos_terms']).keys())
    # # neg_terms = list(collections.Counter(yelp14_all_sample_terms['all_neg_terms']).keys())
    final_output_list = load('viz_log/yelp14_all_{}_sample_terms_doc_data.pkl'.format(model_idx))
    # print(len(final_output_list))
    for key_idx, key in enumerate(pos_keys):
        # if key_idx > 1: break
        # if key != 568: continue
        with open('viz_log/yelp14_all_sample_terms_doc_data_r_234_u_{}.txt'.format(key), 'w') as writer:
            lines = []
            for inst in final_output_list:
                if inst['uid'] == key:
                    pid = inst['pid']
                    doc = inst['doc']
                    rating = inst['rating']
                    if rating in [0, 1, 2, 3, 4]:
                        print('uid:{}, pid:{}'.format(key, pid))
                        lines.append(['uid:{}, pid:{}, rating:{}'.format(key, pid, rating)])
                        lst_words, lst_att_scores = [], []
                        for [sentence, sentence_order, sentence_score] in doc:
                            # top 5 words, ignoring punctuation
                            # if sentence_order > 5: continue
                            words = [word for [word, word_order, word_score] in sentence]
                            scores = [word_score * sentence_score for [word, word_order, word_score] in sentence]

                            lst_words.extend(words)
                            lst_att_scores.extend(scores)

                            # heatmap(words, scores)

                            # punctuation = [',', '.', '!', 'and', 'or', 'the']
                            # sorted_words = sorted(sentence, key=lambda x: x[1])
                            # sorted_words_ignore_punctuation = [w for w in sorted_words if w[0] not in punctuation][0:5]
                            # top_5_words = ['{} (#{})'.format(word, word_order) for [word, word_order, word_score] in sorted_words_ignore_punctuation]
                            # # for [word, word_order, word_score] in sentence:
                            #
                            # print(' '.join(top_5_words))
                            # lines.append(' '.join(top_5_words))

                        list_word_attention_tuple = [(lst_words, lst_att_scores)]
                        generate(list_word_attention_tuple, 'plot/tex/{}/{}/u{}_p{}_r{}.tex'.format(model_idx, key, key, pid, rating))
                        # print('-' * 10)
                        lines.append('-' * 10)
            # print('*' * 100)
            # lines.append('*' * 100)
            # for line in lines:
            #     writer.writelines(line)
            #     writer.writelines('\n')
    print('finished')


                        # for _ in doc_org_data:
                        #     logging.info(_.strip())
                        # label_1 = labels[review_instance_idx]
                        # pred_label = pred_labels[review_instance_idx]

                        # doc_sen_len = review_instance[6]
                        # start_sen_idx, end_sen_idx = tol_sen, tol_sen + len(doc_sen_len)
                        # tol_sen += len(doc_sen_len)
                        # curr_word_score = word_score_list[start_sen_idx:end_sen_idx, :]
                        # # print(np.sum(curr_word_score, axis=-1))
                        # curr_sen_score = sen_score_list[review_instance_idx][0:len(doc_sen_len)]
                        # curr_sen_score_order = np.argsort(-curr_sen_score)

                        # print(np.sum(curr_sen_score, axis=-1))
                        # print()

                        # print(sum(curr_sen_score) -  1)
                        # if user_id != 10:
                        #     continue

                # # find neg and pos
                # pos_terms = []
                # neg_terms = []
                # for sen in doc_org:
                #     for word in sen.split():
                #         polarity = liu_lex_corpus.get(word, None)
                #         if polarity is not None:
                #             if polarity == 'pos':
                #                 pos_terms.append(word)
                #             elif polarity == 'neg':
                #                 neg_terms.append(word)
                #
                #
                # final_output = {'pid': pid, 'uid': uid, 'doc_org': doc_org, 'doc_proc': doc_proc,
                #                 'neg_terms': neg_terms,
                #                 'pos_terms': pos_terms, 'rating': label_0}
                #
                #
                # final_output_list.append(final_output)

    # print(len(final_output_list))
    # print('finished')
    #
    # save(path='viz_log/yelp14_all_{}.pkl'.format(model_idx), obj=final_output_list)
    # obj = load(path='viz_log/yelp14_all_{}.pkl'.format(model_idx))
    # print(len(obj))


    #---
    #
    #
    # print(len(pos_terms))
    #
    # for idx, key in enumerate(pos_keys):
    #     # if idx > 1 : break
    #     for inst in yelp14_all_428:
    #         uid = inst['uid']
    #         pid = inst['pid']
    #         org_doc = inst['doc_org']
    #         doc_proc = inst['doc_proc']
    #         pos_terms = inst['pos_terms']
    #         neg_terms = inst['neg_terms']
    #         rating = inst['rating']
    #
    #         if uid == 568 and rating > 1:
    #             print()
    #
    #
    #     #         # df = pd.DataFrame(data=words_score, index=s_words, columns=['weights'])
    #     #         # ax = sns.heatmap(df, cmap="YlGnBu", xticklabels=False, yticklabels=True)
    #     #         # ax.set_title('{}/{}'.format(s_score_order, len(curr_sen_score_order)))
    #     #         # plt.show()
    #     # for inst in yelp14_all_428:
    #     #     uid = inst['uid']
    #     #     pid = inst['pid']
    #     #     doc_org = inst['doc_org']
    #     #     rating = inst['rating']
    #     #     if uid == key and rating > 1:
    #     #         inst_pos_terms = inst['pos_terms']
    #     #         # inst_neg_terms = inst['neg_terms']
    #     #         for term in inst_pos_terms:
    #     #             if term in pos_terms:
    #     #                 if inst_plot_data.get(term, None) is None:
    #     #                     inst_plot_data[term] = [0, 0, 0]
    #     #
    #     #                 inst_plot_data[term][rating - 2] += 1
    #     #
    #     # words = list(inst_plot_data.keys())
    #     # r2 = [v[0] for (k, v) in inst_plot_data.items()]
    #     # r3 = [v[1] for (k, v) in inst_plot_data.items()]
    #     # r4 = [v[2] for (k, v) in inst_plot_data.items()]
    #     #
    #     # barWidth = 0.35
    #     # xp1 = np.arange(len(r2))
    #     # xp2 = [x + barWidth for x in xp1]
    #     # xp3 = [x + barWidth for x in xp2]
    #     #
    #     # plt.bar(xp1, r2, color='#7f6d5f', width=barWidth, edgecolor='white', label='rating 2')
    #     # plt.bar(xp2, r3, color='#7bd3f7', width=barWidth, edgecolor='white', label='rating 3')
    #     # plt.bar(xp3, r4, color='#2d7f5e', width=barWidth, edgecolor='white', label='rating 4')
    #     #
    #     # plt.xticks([r + barWidth for r in range(len(xp1))], words, rotation=80)
    #     # plt.tick_params(labelsize=8)
    #     #
    #     # plt.xlabel('user #{}'.format(idx), fontweight='light', fontsize=10)
    #     # plt.ylabel('frq.', fontsize=10)
    #     # plt.legend()
    #     # # plot_data = pd.DataFrame.from_dict({'word': list(inst_plot_data.keys()), 'count': list(inst_plot_data.values())})
    #     # # ax = sns .barplot(x=plot_data['word'], y=plot_data['count'], data=plot_data)
    #     # # ax.set_xticklabels(list(plot_data['word']), rotation=70)
    #     # plt.tight_layout()
    #     # plt.savefig('plot/{}_4_.eps'.format(key), format='eps', dpi=1000)
    #     # plt.show()
    #
    # print('finished')
    #
    # # uniform_data = np.random.rand(10, 12)
    # # flights = sns.load_dataset("flights")
    # # flights = flights.pivot("month", "year", "passengers")
    # # print(flights.head())
    # # ax = sns.heatmap(flights)
    # # plt.show()
    #
    # #
    # # df = pd.DataFrame(data=[0.1, 1, 10], index=['a', 'b', 'c'], columns=['weights'])
    # # print(df.head())
    # # ax = sns.heatmap(df, cmap="YlGnBu")
    # # plt.savefig('test.eps', format='eps', dpi=1000)
    # # plt.show()
    # #     if uid == 1:
    # #
    # #         if rating_pos.get(rating, None) is None:
    # #             rating_pos[rating] = []
    # #
    # #         rating_pos[rating].extend(inst['pos_terms'])
    # # #
    # # # for k, v in rating_pos.items():
    # # #     v_ = collections.Counter(v)
    # # #     print(k)
    # # #     print(len(v_.keys()))
    # # #     print(v_)
    # #
    # # # pos words only in 4
    # # r0 = list(collections.Counter(rating_pos.get(0)).keys())
    # # r1 = list(collections.Counter(rating_pos.get(1)).keys())
    # # r2 = list(collections.Counter(rating_pos.get(2)).keys())
    # # r3 = list(collections.Counter(rating_pos.get(3)).keys())
    # # r4 = list(collections.Counter(rating_pos.get(4)).keys())
    # #
    # # r0_3 = r0 + r1 + r2 + r3
    # # for k in r4:
    # #     if k not in r0_3:
    # #         print(k, collections.Counter(rating_pos.get(4))[k])
    # #
    #
    #
    # # pos words only in 3