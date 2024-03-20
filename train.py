import models 
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.data_loader import get_loader
from fewshot_re_kit.sentence_encoder import BERTSentenceEncoder

import torch, os, json, argparse, wandb, sys
import numpy as np

def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('--train', default='fewrel/train_wiki',
                help='train file')
        parser.add_argument('--val', default='fewrel/val_wiki',
                help='val file')
        parser.add_argument('--test', default='fewrel/val_wiki',
                help='test file')
        parser.add_argument('--adv', default=None,
                help='adv file')
        parser.add_argument('--trainN', default=10, type=int,
                help='N in train')
        parser.add_argument('--N', default=5, type=int,
                help='N way')
        parser.add_argument('--K', default=5, type=int,
                help='K shot')
        parser.add_argument('--Q', default=5, type=int,
                help='Num of query per class')
        parser.add_argument('--batch_size', default=4, type=int,
                help='batch size')
        parser.add_argument('--train_iter', default=20, type=int,
                help='num of iters in training')
        parser.add_argument('--val_iter', default=10, type=int,
                help='num of iters in validation')
        parser.add_argument('--test_iter', default=10, type=int,
                help='num of iters in testing')
        parser.add_argument('--val_step', default=5, type=int,
                help='val after training how many iters')
        parser.add_argument('--model', default='proto',
                help='model name')
        parser.add_argument('--encoder', default='bert',
                help='encoder: cnn or bert or roberta')
        parser.add_argument('--max_length', default=128, type=int,
                help='max length')
        parser.add_argument('--lr', default=2e-5, type=float,
                help='learning rate')
        parser.add_argument('--weight_decay', default=1e-5, type=float,
                help='weight decay')
        parser.add_argument('--dropout', default=0.0, type=float,
                help='dropout rate')
        parser.add_argument('--na_rate', default=0, type=int,
                help='NA rate (NA = Q * na_rate)')
        parser.add_argument('--grad_iter', default=1, type=int,
                help='accumulate gradient every x iterations')
        parser.add_argument('--hidden_size', default=230, type=int,
                help='hidden size')
        parser.add_argument('--load_ckpt', default=None,
                help='load ckpt')
        parser.add_argument('--save_ckpt', default=None,
                help='save ckpt')
        parser.add_argument('--fp16', action='store_true',
                help='use nvidia apex fp16')
        parser.add_argument('--only_test', action='store_true',
                help='only test')
        parser.add_argument('--ckpt_name', type=str, default='',
                help='checkpoint name.')
        parser.add_argument('--pretrain_ckpt', default=None,
                help='bert / roberta pre-trained checkpoint')
        parser.add_argument('--cat_entity_rep', action='store_true', help='concatenate entity representation as sentence rep')

        # multirep args
        parser.add_argument('--add_loss_rdcl', action='store_true')
        parser.add_argument('--desc_max_length', default=32, type=int)

        # representations
        parser.add_argument('--remove_cls', action='store_true')
        parser.add_argument('--remove_mean', action='store_true')
        parser.add_argument('--remove_mask', action='store_true')
        parser.add_argument('--remove_entitymarker', action='store_true')

        opt = parser.parse_args()

        trainN = opt.trainN
        N = opt.N
        K = opt.K
        Q = opt.Q
        loss_rdcl = int(opt.add_loss_rdcl)
        batch_size = opt.batch_size
        model_name = opt.model
        encoder_name = opt.encoder
        max_length = opt.max_length
        opt.train_iter = opt.train_iter * opt.grad_iter
        opt.val_step = opt.val_step * opt.grad_iter

        add_cls = not opt.remove_cls
        add_mean = not opt.remove_mean
        add_mask = not opt.remove_mask
        add_entitymarker = not opt.remove_entitymarker

        if opt.encoder == 'bert':
                pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'
                sentence_encoder = BERTSentenceEncoder(
                pretrain_ckpt,
                max_length,
                cat_entity_rep=opt.cat_entity_rep,
                multirep=model_name == "multirep",
                desc=opt.add_loss_rdcl,
                desc_max_length=opt.desc_max_length,
                )
        else:
                raise NotImplementedError
    
        train_data_loader = get_loader(opt.train, sentence_encoder,
                        N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, n_iter=opt.train_iter*batch_size)
        val_data_loader = get_loader(opt.val, sentence_encoder,
                N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, n_iter=opt.val_iter*batch_size)
        test_data_loader = get_loader(opt.test, sentence_encoder,
                N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, n_iter=opt.test_iter*batch_size)
        
        optim = torch.optim.AdamW
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')

        framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader, device=device)

        prefix = '-'.join([model_name, encoder_name, opt.train.split("/")[0], opt.val.split("/")[0], str(N), str(K)])
        if opt.adv is not None:
                prefix += '-adv_' + opt.adv
        if opt.na_rate != 0:
                prefix += '-na{}'.format(opt.na_rate)
        if opt.cat_entity_rep:
                prefix += '-catentity'
        if len(opt.ckpt_name) > 0:
                prefix += '-' + opt.ckpt_name
        if opt.add_loss_rdcl:
                prefix += '-rdcl'
        
        # Identify different model variants
        if add_cls:
                prefix += '-CLS'
        if add_mask:
                prefix += '-MASK'
        if add_mean:
                prefix += '-MEAN'
        if add_entitymarker:
                prefix += '-ENTITYMARKER'

        if model_name == "multirep":
                if opt.add_loss_rdcl:
                        model = models.multirep.MultiRepTD(
                                sentence_encoder,
                                add_cls=add_cls,
                                add_mean=add_mean,
                                add_mask=add_mask,
                                add_entitymarker=add_entitymarker,
                                )
                else:
                        model = models.multirep.MultiRep(
                                sentence_encoder,
                                add_cls=add_cls,
                                add_mean=add_mean,
                                add_mask=add_mask,
                                add_entitymarker=add_entitymarker,
                                )

        if not os.path.exists('checkpoint'):
                os.mkdir('checkpoint')
        ckpt = 'checkpoint/{}.bin'.format(prefix)
        if opt.save_ckpt:
                ckpt = opt.save_ckpt

        model.to(device)

        # wandb init
        wandb.init(
                project="MultiRep_RR2",
                config=vars(opt),
                name=prefix,
        )

        if not opt.only_test:
                framework.train(model, prefix, batch_size, trainN, N, K, Q,
                        pytorch_optim=optim, load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                        na_rate=opt.na_rate, val_step=opt.val_step, fp16=opt.fp16, 
                        train_iter=opt.train_iter, val_iter=opt.val_iter, 
                        learning_rate=opt.lr, grad_iter=opt.grad_iter)

        # model.load_state_dict(torch.load(ckpt))        
        acc = framework.eval(model, batch_size, N, K, Q, opt.test_iter, na_rate=opt.na_rate, ckpt=ckpt)
        print("RESULT: %.2f" % (acc * 100))

if __name__ == "__main__":
    main()
