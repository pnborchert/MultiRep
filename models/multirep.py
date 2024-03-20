import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

class MultiRep(fewshot_re_kit.framework.FewShotREModel):
    """MultiRep model"""
    def __init__(
            self,
            sentence_encoder,
            dropout=0.2,
            add_cls=True,
            add_mask=True,
            add_mean=True,
            add_entitymarker=True,
            **kwargs
            ):
        
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder, **kwargs)
        self.hidden_size = sentence_encoder.bert.config.hidden_size
        self.add_cls = add_cls
        self.add_mask = add_mask
        self.add_mean = add_mean
        self.add_entitymarker = add_entitymarker
        self.n_rep = sum([add_cls, add_mask, add_mean, add_entitymarker*2])

    def rcl_loss(self, x, temp=0.05):
        """Representation-Representation Contrastive Loss"""
        B,N,K,R,D = x.shape
        x = x.view(B,N*K*R,D)
        loss = 0
        for i in range(B):
            xi = F.normalize(x[i], dim=1)
            sim = xi @ xi.transpose(0,1)
            sim = torch.exp(sim / temp)
            mask = torch.as_tensor([[i for _ in range(K*R)] for i in range(N)]).reshape(-1).to(x.device)
            mask = torch.eq(mask, mask.unsqueeze(1)).float() - torch.eye(mask.shape[0]).to(x.device)
            pos = (sim * mask).sum(-1)
            neg = (sim.sum(-1) - sim.diag())
            loss += -torch.log(pos / (pos + neg)).sum()
        return loss / (B*N*K*R)
    
    def __dist__(self, x,y, dim):
        return (x*y).sum(dim=dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)
    
    def get_index(self, x, index_tensor):
        i = torch.arange(x.shape[0])
        return x[i, index_tensor]
    
    def forward(self, support, query, N, K, total_Q, is_eval=False):
        support_emb = self.sentence_encoder(support)
        s_h = support_emb.last_hidden_state
        s_cls = self.get_index(s_h, torch.zeros_like(support["pos_mask"]))
        s_mask = self.get_index(s_h, support["pos_mask"])
        s_e1 = self.get_index(s_h, support["pos1"])
        s_e2 = self.get_index(s_h, support["pos2"])

        query_emb = self.sentence_encoder(query)
        q_h = query_emb.last_hidden_state
        q_cls = self.get_index(q_h, torch.zeros_like(query["pos_mask"]))
        q_mask = self.get_index(q_h, query["pos_mask"])
        q_e1 = self.get_index(q_h, query["pos1"])
        q_e2 = self.get_index(q_h, query["pos2"])

        s_rep_list = []
        q_rep_list = []
        if self.add_cls:
            s_rep_list.append(s_cls)
            q_rep_list.append(q_cls)
        if self.add_mask:
            s_rep_list.append(s_mask)
            q_rep_list.append(q_mask)
        if self.add_mean:
            s_rep_list.append(s_h.mean(1))
            q_rep_list.append(q_h.mean(1))
        if self.add_entitymarker:
            s_rep_list.append(s_e1)
            s_rep_list.append(s_e2)
            q_rep_list.append(q_e1)
            q_rep_list.append(q_e2)

        s_rep = torch.cat(s_rep_list, dim=-1).reshape(-1, N, K, self.n_rep, self.hidden_size)
        q_rep = torch.cat(q_rep_list, dim=-1).reshape(-1, total_Q, self.n_rep*self.hidden_size)

        # query instance to prototype logits
        proto = s_rep.view(-1, N, K, self.n_rep*self.hidden_size).mean(2)
        logits = self.__batch_dist__(proto, q_rep)
        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2).view(-1, N + 1)
        _, pred = torch.max(logits, 1)

        if not is_eval:
            # representation to representation contrastive loss
            loss_rcl = self.rcl_loss(s_rep)

            return logits, pred, loss_rcl, torch.tensor(0.0)
        
        else:
            return logits, pred, None, None

class MultiRepTD(fewshot_re_kit.framework.FewShotREModel):
    """MultiRep model incorporating textual relation descriptions"""
    def __init__(
            self,
            sentence_encoder,
            dropout=0.2,
            add_cls=True,
            add_mask=True,
            add_mean=True,
            add_entitymarker=True,
            **kwargs
            ):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder, **kwargs)
        self.hidden_size = sentence_encoder.bert.config.hidden_size
        
        self.add_cls = add_cls
        self.add_mask = add_mask
        self.add_mean = add_mean
        self.add_entitymarker = add_entitymarker
        self.n_rep = sum([add_cls, add_mask, add_mean, add_entitymarker*2])
    
    def rdcl_loss(self, x, d, temp=0.05):
        """Instance-Relation Description Contrastive Loss"""
        B,N,K,R,D = x.shape
        x = x.view(B,N*K,R*D)
        d = d.view(B,N,R*D)
        loss = 0
        for i in range(B):
            xi = F.normalize(x[i], dim=1)
            di = F.normalize(d[i], dim=1)
            sim = xi @ di.transpose(0,1)
            sim = torch.exp(sim /temp)
            mask = torch.zeros((N*K,N), dtype=bool)
            for i in range(N):
                mask[i*K:(i+1)*K,i] = True
            pos = sim[mask]
            neg = (sim.sum(1) - pos)
            loss += -torch.log(pos / (pos + neg)).sum()
        return loss / (B*N*K)

    def rcl_loss(self, x, temp=0.05):
        """Representation-Representation Contrastive Loss"""
        B,N,K,R,D = x.shape
        x = x.view(B,N*K*R,D)
        loss = 0
        for i in range(B):
            xi = F.normalize(x[i], dim=1)
            sim = xi @ xi.transpose(0,1)
            sim = torch.exp(sim / temp)
            mask = torch.as_tensor([[i for _ in range(K*R)] for i in range(N)]).reshape(-1).to(x.device)
            mask = torch.eq(mask, mask.unsqueeze(1)).float() - torch.eye(mask.shape[0]).to(x.device)
            pos = (sim * mask).sum(-1)
            neg = (sim.sum(-1) - sim.diag())
            loss += -torch.log(pos / (pos + neg)).sum()
        return loss / (B*N*K*R)
    
    def __dist__(self, x,y, dim):
        return (x*y).sum(dim=dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)
    
    def get_index(self, x, index_tensor):
        i = torch.arange(x.shape[0])
        return x[i, index_tensor]
    
    def forward(self, support, query, N, K, total_Q, is_eval=False):
        support_emb, support_desc_emb = self.sentence_encoder(support)
        s_h = support_emb.last_hidden_state
        s_cls = self.get_index(s_h, torch.zeros_like(support["pos_mask"]))
        s_mask = self.get_index(s_h, support["pos_mask"])
        s_e1 = self.get_index(s_h, support["pos1"])
        s_e2 = self.get_index(s_h, support["pos2"])
        
        query_emb, query_desc_emb = self.sentence_encoder(query)
        q_h = query_emb.last_hidden_state
        q_cls = self.get_index(q_h, torch.zeros_like(query["pos_mask"]))
        q_mask = self.get_index(q_h, query["pos_mask"])
        q_e1 = self.get_index(q_h, query["pos1"])
        q_e2 = self.get_index(q_h, query["pos2"])

        s_rep_list = []
        q_rep_list = []
        if self.add_cls:
            s_rep_list.append(s_cls)
            q_rep_list.append(q_cls)
        if self.add_mask:
            s_rep_list.append(s_mask)
            q_rep_list.append(q_mask)
        if self.add_mean:
            s_rep_list.append(s_h.mean(1))
            q_rep_list.append(q_h.mean(1))
        if self.add_entitymarker:
            s_rep_list.append(s_e1)
            s_rep_list.append(s_e2)
            q_rep_list.append(q_e1)
            q_rep_list.append(q_e2)

        s_rep = torch.cat(s_rep_list, dim=-1).reshape(-1, N, K, self.n_rep, self.hidden_size)
        q_rep = torch.cat(q_rep_list, dim=-1).reshape(-1, total_Q, self.n_rep*self.hidden_size)

        # description representation
        s_desc_cls = self.get_index(support_desc_emb.last_hidden_state, torch.zeros_like(support["desc_pos_mask"]))
        s_desc_mask =  self.get_index(support_desc_emb.last_hidden_state, support["desc_pos_mask"])
        s_desc_pool = F.dropout(support_desc_emb.pooler_output, p=0.1, training=True)
        s_desc_mask_drop10 = F.dropout(
            self.get_index(support_desc_emb.last_hidden_state, torch.zeros_like(support["desc_pos_mask"])),
            p=0.1,
            training=True)
        
        s_desc_rep_list = []
        if self.add_cls:
            s_desc_rep_list.append(s_desc_cls)
        if self.add_mask:
            s_desc_rep_list.append(s_desc_mask)
        if self.add_mean:
            s_desc_rep_list.append(support_desc_emb.last_hidden_state.mean(1))
        if self.add_entitymarker:
            s_desc_rep_list.append(s_desc_pool)
            s_desc_rep_list.append(s_desc_mask_drop10)
        
        s_desc_rep = torch.cat(s_desc_rep_list, dim=-1).reshape(-1, N, self.n_rep*self.hidden_size)

        proto = s_rep.view(-1, N, K, self.n_rep*self.hidden_size).mean(2)

        proto += s_desc_rep
        logits = self.__batch_dist__(proto, q_rep)

        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2).view(-1, N + 1)
        _, pred = torch.max(logits, 1)

        if not is_eval:
            
            # representation to representation contrastive loss
            loss_rcl = self.rcl_loss(s_rep)

            # representation to description contrastive loss
            loss_rdcl = self.rdcl_loss(s_rep, s_desc_rep)

            return logits, pred, loss_rcl, loss_rdcl
        
        else:
            return logits, pred, None, None
