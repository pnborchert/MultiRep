import os
import sklearn.metrics
import numpy as np
import sys
import time
import wandb
from . import sentence_encoder
from . import data_loader
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
# from pytorch_pretrained_bert import BertAdam
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm,trange

def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return global_step / warmup_step
    else:
        return 1.0

class FewShotREModel(nn.Module):
    def __init__(self, my_sentence_encoder):
        '''
        sentence_encoder: Sentence encoder
        
        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        self.sentence_encoder = my_sentence_encoder
        self.cost = nn.CrossEntropyLoss()
    
    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits, pred
        '''
        raise NotImplementedError

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size. 
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))

class FewShotREFramework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader, adv_data_loader=None, adv=False, d=None, device="cuda"):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.adv_data_loader = adv_data_loader
        self.adv = adv
        self.device = device
    
    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)
    
    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def train(self,
              model,
              model_name,
              B, N_for_train, N_for_eval, K, Q,
              na_rate=0,
              learning_rate=1e-1,
              lr_step_size=20000,
              weight_decay=1e-5,
              train_iter=30000,
              val_iter=1000,
              val_step=2000,
              test_iter=3000,
              load_ckpt=None,
              save_ckpt=None,
              pytorch_optim=optim.SGD,
              bert_optim=False,
              warmup=True,
              warmup_step=300,
              grad_iter=1,
              fp16=False,
              pair=False,
              adv_dis_lr=1e-1,
              adv_enc_lr=1e-1,
              use_sgd_for_bert=False,
              ):
        '''
        model: a FewShotREModel instance
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        test_iter: Num of iterations of testing
        '''
        print("Start training...")
    
        # Init
        if bert_optim:
            print('Use bert optim!')
            parameters_to_optimize = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            parameters_to_optimize = [
                {'params': [p for n, p in parameters_to_optimize 
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in parameters_to_optimize
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            if use_sgd_for_bert:
                optimizer = torch.optim.SGD(parameters_to_optimize, lr=learning_rate)
            else:
                optimizer = AdamW(parameters_to_optimize, lr=learning_rate, correct_bias=False)
            if self.adv:
                optimizer_encoder = AdamW(parameters_to_optimize, lr=1e-5, correct_bias=False)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter) 
        else:
            optimizer = pytorch_optim(model.parameters(),
                    learning_rate, weight_decay=weight_decay)
            if self.adv:
                optimizer_encoder = pytorch_optim(model.parameters(), lr=adv_enc_lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)

        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    print('ignore {}'.format(name))
                    continue
                print('load {} from {}'.format(name, load_ckpt))
                own_state[name].copy_(param)
            start_iter = 0
        else:
            start_iter = 0

        if fp16:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        model.train()

        # Training
        best_acc = 0
        iter_loss = 0.0
        iter_ce_loss = 0.0
        iter_loss_rcl = 0.0
        iter_loss_rdcl = 0.0
        iter_right = 0.0
        iter_sample = 0.0

        dl = iter(self.train_data_loader)

        pbar = trange(start_iter, start_iter + train_iter, desc="Train", leave=True)
        for it in pbar:
                
            support, query, label = next(dl)
            # support, query, label, target_classes = next(self.train_data_loader)
            for k in support:
                support[k] = support[k].to(self.device)
            for k in query:
                query[k] = query[k].to(self.device)
            label = label.to(self.device)
            # target_classes = target_classes.to(self.device)

            logits, pred, loss_rcl, loss_rdcl = model(support, query, 
                    N_for_train, K, Q * N_for_train + na_rate * Q)
            ce_loss = model.loss(logits, label) / float(grad_iter)
            loss = ce_loss + loss_rcl + loss_rdcl
            right = model.accuracy(pred, label)
            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 10)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

            iter_loss += loss.item()
            iter_right += right.item()
            iter_ce_loss += ce_loss.item()
            iter_loss_rcl += loss_rcl.item()
            iter_loss_rdcl += loss_rdcl.item()
            iter_sample += 1

            metrics = {
                'loss': iter_loss / iter_sample,
                'accuracy': iter_right / iter_sample,
                "ce_loss": iter_ce_loss / iter_sample,
                "loss_rcl": iter_loss_rcl / iter_sample,
                "loss_rdcl": iter_loss_rdcl / iter_sample,
                }
            
            for k,v in metrics.items():
                wandb.log({f"train/{k}": v, "step": it})
            pbar.set_postfix({k: "{0:1.5f}".format(v) for k, v in metrics.items()})
            
            
            if (it + 1) % grad_iter == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (it + 1) % val_step == 0:
                acc = self.eval(model, B, N_for_eval, K, Q, val_iter, na_rate=na_rate, pair=pair, train_step=it)
                model.train()
                if acc > best_acc:
                    print('Best checkpoint')
                    torch.save(model, save_ckpt)
                    best_acc = acc
                iter_loss = 0.
                iter_ce_loss = 0.
                iter_loss_rcl = 0.
                iter_loss_rdcl = 0.
                iter_right = 0.
                iter_sample = 0.
                
        print("\n####################\n")
        print("Finish training " + model_name)

    def eval(self,
            model,
            B, N, K, Q,
            eval_iter,
            na_rate=0,
            pair=False,
            ckpt=None,
            train_step=None,
            ): 
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("")
        
        model.eval()
        if ckpt is None:
            print("Use val dataset")
            dl = iter(self.val_data_loader)
            mode = "val"
        else:
            print("Use test dataset")
            if ckpt != 'none':
                print("load checkpoint: " + ckpt)
                model.load_state_dict(torch.load(ckpt).state_dict())
                # state_dict = self.__load_model__(ckpt)['state_dict']
                # own_state = model.state_dict()
                # for name, param in state_dict.items():
                #     if name not in own_state:
                #         continue
                #     own_state[name].copy_(param)
            dl = iter(self.test_data_loader)
            mode = "test"

        iter_right = 0.0
        iter_sample = 0.0
        iter_ce_loss = 0.0
        
        with torch.no_grad():
            pbar = trange(eval_iter, desc="Eval", leave=True)
            for it in pbar:
                support, query, label = next(dl)
                # support, query, label, target_classes = next(eval_dataset)
                for k in support:
                    support[k] = support[k].to(self.device)
                for k in query:
                    query[k] = query[k].to(self.device)
                label = label.to(self.device)
                # target_classes = target_classes.to(self.device)
                logits, pred, _, _ = model(support, query, N, K, Q * N + Q * na_rate, is_eval=True)

                ce_loss = model.loss(logits, label)
                right = model.accuracy(pred, label)
                iter_right += self.item(right.data)
                iter_ce_loss += self.item(ce_loss.data)
                iter_sample += 1

                metrics = {
                    'accuracy': iter_right / iter_sample,
                    "ce_loss": iter_ce_loss / iter_sample,
                }

                pbar.set_postfix({k: "{0:1.5f}".format(v) for k, v in metrics.items()})
        
        for k,v in metrics.items():
            if mode == "val":
                wandb.log({f"{mode}/{k}": v, "step": train_step})
            else:
                wandb.log({f"{mode}/{k}": v})
                
        return iter_right / iter_sample
