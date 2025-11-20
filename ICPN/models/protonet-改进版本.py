import torch
import numpy as np
import torch.nn.functional as F
num_eval_episodes
num_test_episodes
from models import FewShotModel

# Note: As in Protonet, we use Euclidean Distances here, you can change to the Cosine Similarity by replace
#       TRUE in line 30 as self.args.use_euclidean

class ProtoNet(FewShotModel):
    def __init__(self, args):
        super().__init__(args)

    def _forward(self, instance_embs, support_idx, query_idx):
        #instance_embs是一个所有特征向量的嵌入表示
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))

        query   = instance_embs[query_idx.flatten()].view(  *(query_idx.shape   + (-1,)))

        # get mean of the support
        # #得到原型均值
        # proto = support.mean(dim=1) # Ntask x NK x d

        weights = torch.rand_like(support).to(support.device)
        # weights = ProtoNet.generate_random_weights(10)
        proto = (support * weights).sum(dim=1) / weights.sum(dim=1)

        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])

        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        if True: # self.args.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
            proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim)
            proto = proto.contiguous().view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)

            logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature
        else: # cosine similarity: more memory efficient
            proto = F.normalize(proto, dim=-1) # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)

            # (num_batch,  num_emb, num_proto) * (num_batch, num_query*num_proto, num_emb) -> (num_batch, num_query*num_proto, num_proto)
            logits = torch.bmm(query, proto.permute([0,2,1])) / self.args.temperature
            logits = logits.view(-1, num_proto)

        if self.training:
            return logits, None
        else:
            return logits

    def generate_random_weights(size):
        upper_bound = size * 10

        values_to_pick_from = list(range(1, upper_bound + 1))
        values_to_pick_from = np.array(values_to_pick_from)

        chosen_values = np.random.choice(values_to_pick_from, size, replace = False)
        chosen_values_sum = sum(chosen_values)

        return [(x / chosen_values_sum) for x in chosen_values]
