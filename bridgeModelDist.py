# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from bi_lstm_dist import bi_lstm_dist
# from bi_lstm_trans_gmm_joint import bi_lstm_trans_gmm_joint
# from AbsaOptim import ScheduledOptim


class bridgeModelDist(nn.Module):
    def __init__(self,
                 model_name,
                 dim_word_input, dim_sen_hidden, dim_doc_hidden,
                 dim_user_product_input, dim_user_product_hidden,
                 dim_user_doc_hidden, dim_product_doc_hidden, dim_user_product_doc_hidden,

                 init_usr_centers, init_pdr_centers, num_cluster,
                 usr_rate_dist, pdr_rate_dist,
                 doc_embed, usr_embed, pdr_embed,

                 learning_rate, lr_word_vector, weight_decay=0,
                 batch_size=32, max_length_sen=50,
                 embed_dropout_rate=0., cell_dropout_rate=0., final_dropout_rate=0.,
                 n_layers=2, n_classes=10, bidirectional=True,
                 optim_type="Adam", rnn_type="LSTM", lambda1=0.001,
                 use_cuda=True):

        super(bridgeModelDist, self).__init__()

        self.model_name = model_name
        self.n_class = n_classes
        self.max_length_sen = max_length_sen
        self.batch_size = batch_size

        self.model = bi_lstm_dist(dim_word_input=dim_word_input, dim_sentence_hidden=dim_sen_hidden, dim_doc_hidden=dim_doc_hidden,
                                  dim_user_product_input=dim_user_product_input, dim_user_product_hidden=dim_user_product_hidden,
                                  dim_user_doc_hidden=dim_user_doc_hidden, dim_product_doc_hidden=dim_product_doc_hidden, dim_user_product_doc_hidden=dim_user_product_doc_hidden,
                                  init_usr_centers=init_usr_centers, init_pdr_centers=init_pdr_centers, num_cluster=num_cluster,
                                  doc_embed_list=doc_embed, usr_embed_list=usr_embed, pdr_embed_list=pdr_embed,
                                  usr_rate_dist=usr_rate_dist, pdr_rate_dist=pdr_rate_dist,
                                  n_layers=n_layers, n_classes=n_classes,
                                  embed_dropout_rate=embed_dropout_rate, cell_dropout_rate=cell_dropout_rate,
                                  bidirectional=bidirectional, rnn_type=rnn_type, use_cuda=use_cuda)

        # verify default value
        self.optimizer = getattr(optim, optim_type)([
            {'params': self.model.base_params, 'name': 'base_params'},
            {'params': self.model.doc_embed.parameters(), 'lr': lr_word_vector, 'weight_decay': 0, 'name': 'doc_embed'},
            {'params': self.model.usr_embed.parameters(), 'lr': lr_word_vector, 'weight_decay': 0, 'name': 'usr_embed'},
            {'params': self.model.pdr_embed.parameters(), 'lr': lr_word_vector, 'weight_decay': 0, 'name': 'pdr_embed'}],
            lr=learning_rate, weight_decay=weight_decay, amsgrad=True
            )

        # doesn't work
        # self.scheduled_optimizer = ScheduledOptim(optimizer=self.optimizer, d_model=512, n_warmup_steps=40)

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model.cuda()

        # self.optimizer = getattr(optim, optim_type)([
        #     {'params': self.model.base_params},
        #     {'params': self.model.doc_embed.parameters(), 'lr': lr_word_vector, 'weight_decay': 0}
        #     # {'params': self.model.usr_embed.parameters(), 'lr': lr_word_vector, 'weight_decay': 0},
        #     # {'params': self.model.pdr_embed.parameters(), 'lr': lr_word_vector, 'weight_decay': 0}
        # ], lr=self.learning_rate, weight_decay=weight_decay)

    def get_batch_data(self, batched_data):
        # usr, prd, docs, label, sen_len, doc_len
        # batch_size * num of sentence * num of words
        batched_docs = []
        doc_labels, docs_len, sens_len = [], [], []
        usrs, pdrs = [], []
        usrs_rate_dist_list, pdrs_rate_dist_list = [], []
        very_small_value = 1e-12
        #usr, prd, t_usr_rate_dist, t_pdr_rate_dist, label, docs, docs_len
        for (usr, prd, usr_rate_dist, pdr_rate_dist, label, doc, lens) in batched_data:
            batched_docs.extend(doc)
            docs_len.append(len(lens))
            sens_len.extend(lens)
            doc_labels.append(label)
            usrs.append(usr)
            usrs_rate_dist_list.append(usr_rate_dist)
            usr_rate_dist[usr_rate_dist == 0] = very_small_value # replace 0 with very small value to avoid kl explosure
            pdr_rate_dist[pdr_rate_dist == 0] = very_small_value
            pdrs.append(prd)
            pdrs_rate_dist_list.append(pdr_rate_dist)

        var_batched_docs = Variable(torch.LongTensor(batched_docs)).cuda() if self.use_cuda else Variable(torch.LongTensor(batched_docs))
        var_doc_labels = Variable(torch.LongTensor(doc_labels)).cuda() if self.use_cuda else Variable(torch.LongTensor(doc_labels))
        var_usrs = Variable(torch.LongTensor(usrs)).cuda() if self.use_cuda else Variable(torch.LongTensor(usrs))
        var_pdrs = Variable(torch.LongTensor(pdrs)).cuda() if self.use_cuda else Variable(torch.LongTensor(pdrs))
        usr_rate_dist = Variable(torch.FloatTensor(usrs_rate_dist_list)).cuda() if self.use_cuda else Variable(torch.FloatTensor(usrs_rate_dist_list))
        pdr_rate_dist = Variable(torch.FloatTensor(pdrs_rate_dist_list)).cuda() if self.use_cuda else Variable(torch.FloatTensor(pdrs_rate_dist_list))

        dict_data = {'docs': var_batched_docs, 'usrs': var_usrs, 'prds': var_pdrs, 'usr_rate_dist': usr_rate_dist, 'pdr_rate_dist': pdr_rate_dist,
                     'labels': var_doc_labels, 'len_docs': docs_len, 'len_sens': sens_len}

        return dict_data

    def predict(self, batched_data):
        # Turn on training mode which enables dropout.
        self.model.eval()

        # with response instances
        b_data = self.get_batch_data(batched_data)
        pdr_usr_prob = self.model(b_data)
        loss = F.nll_loss(pdr_usr_prob, b_data['labels'], ignore_index=-1)
        return pdr_usr_prob.data.cpu().numpy(), loss.data.cpu().numpy()

    def stepTrain(self, batched_data):
        # Turn on training mode which enables dropout.
        self.model.train()
        self.optimizer.zero_grad()
        # self.scheduled_optimizer.zero_grad()

        b_data = self.get_batch_data(batched_data)

        pdr_usr_prob = self.model(b_data)
        loss = F.nll_loss(pdr_usr_prob, b_data['labels'], ignore_index=-1)
        loss.backward()

        self.optimizer.step()
        return pdr_usr_prob.data.cpu().numpy(), loss.data.cpu().numpy()

    def save_model(self, dir_path, idx):
        os.mkdir(dir_path) if not os.path.isdir(dir_path) else None
        torch.save(self.state_dict(), '%s/model%s.pth' % (dir_path, idx))
        # torch.save(self, '%s/model%s.pkl' % (dir, idx))

    def load_model(self, dir_path, idx=-1):
        if idx < 0:
            params = torch.load('%s' % dir_path)
            self.load_state_dict(params)
        else:
            self.load_state_dict(torch.load('%s/model%s.pth' % (dir_path, idx)))