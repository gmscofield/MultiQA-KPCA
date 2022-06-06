import torch
import torch.nn as nn
from transformers import BertModel,BertTokenizer
import numpy as np
from scipy.spatial.distance import pdist, squareform


def rbf(x, gamma=15):
    sq_dists = pdist(x, 'sqeuclidean')
    mat_sq_dists = squareform(sq_dists)
    return np.exp(-gamma * mat_sq_dists)


class KPCA():
    def __init__(self,kernel=rbf):
        super(KPCA,self).__init__()
        self.kernel=kernel

    def kpca(self,data,n_dims=2):
        K = self.kernel(data)
        N = K.shape[0]
        one_n = np.ones((N, N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
        eig_values, eig_vector = np.linalg.eig(K)
        idx = eig_values.argsort()[::-1]
        eigval = eig_values[idx][:n_dims]
        eigvector = eig_vector[:, idx][:, :n_dims]
        eigval = eigval ** (1 / 2)
        vi = eigvector / eigval.reshape(-1, n_dims)
        data_n = np.dot(K, vi)
        return data_n

class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tag_linear = nn.Linear(5,self.bert.config.hidden_size)
        self.kpca=KPCA()
        self.linear = nn.Linear(1,5)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.loss_func = nn.CrossEntropyLoss()
        self.theta = config.theta

    def forward(self, input, attention_mask, token_type_ids, context_mask=None, turn_mask=None, target_tags=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rep, _ = self.bert(input, attention_mask, token_type_ids)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        rep = torch.tensor([tokenizer.encode(rep, add_special_tokens=False)]).float()
        # print(rep)
        rep = self.dropout(rep)
        rep=rep.to(device)
        tag_logits = self.tag_linear(rep)  # (batch,seq_len,num_tag)
        # tem=tag_logits.detach().numpy().T
        tem=tag_logits.cpu().detach().numpy().T
        tem = self.kpca.kpca(data=tem,n_dims=168)
        tag_logits = torch.from_numpy(np.float32(tem))
        tag_logits = tag_logits.to(device)
        w=torch.randn((10,tag_logits.size()[0])).to(device)
        tag_logits = torch.mm(w,tag_logits)
        tag_logits = tag_logits.view((tag_logits.size()[0],tag_logits.size()[1],1))
        tag_logits = self.linear(tag_logits)

        if not target_tags is None:
            tag_logits_t1 = tag_logits[turn_mask == 0]  # (n1,seq_len,num_tag)
            target_tags_t1 = target_tags[turn_mask == 0]  # (n1,seq_len)
            context_mask_t1 = context_mask[turn_mask == 0]  # (n1,seq_len)

            tag_logits_t2 = tag_logits[turn_mask == 1]  # (n2,seq_len,num_tag)
            target_tags_t2 = target_tags[turn_mask == 1]  # (n2,seq_len)
            context_mask_t2 = context_mask[turn_mask == 1]  # (n2,seq_len)

            tag_logits_t1 = tag_logits_t1[context_mask_t1 == 1]  # (N1,num_tag)
            target_tags_t1 = target_tags_t1[context_mask_t1 == 1]  # (N1)

            tag_logits_t2 = tag_logits_t2[context_mask_t2 == 1]  # (N2,num_tag)
            target_tags_t2 = target_tags_t2[context_mask_t2 == 1]  # (N2)

            loss_t1 = self.loss_func(tag_logits_t1, target_tags_t1) if len(
                target_tags_t1) != 0 else torch.tensor(0).type_as(input)
            loss_t2 = self.loss_func(tag_logits_t2, target_tags_t2) if len(
                target_tags_t2) != 0 else torch.tensor(0).type_as(input)
            loss = self.theta*loss_t1+(1-self.theta)*loss_t2
            return loss, (loss_t1.item(), loss_t2.item())
        else:
            # for prediction
            tag_idxs = torch.argmax(
                tag_logits, dim=-1).squeeze(-1)  # (batch,seq_len)
            return tag_idxs
