"""
 基于混合图谱的对话推荐系统
 社交网络挖掘
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse
from tqdm import tqdm
import pickle as pkl
from nltk.translate.bleu_score import sentence_bleu
import torch
from torch import optim
from mainModel import KGSFModel
from dataset import dataset, CRSdataset

class TrainModel():
    def __init__(self, opt):
        self.opt = opt
        self.trainset = dataset("data/train_data.jsonl", opt)
        self.movie_id = pkl.load(open("data/movie_ids.pkl", "rb"))
        self.dict = self.trainset.word2index
        self.index2word = {self.dict[word]: word for word in self.dict}
        self.epoch = opt['epoch']
        self.batchsize = opt['batch_size']
        self.is_finetune = opt['is_finetune']
        self.on_cuda = opt['on_cuda']
        self.rec_accuracy = {'cnt':0, 'top1_acc':0, 'top10_acc':0, 'top50_acc':0, 'loss_sum':0}
        self.res_accuracy = {'cnt': 0, 'bleu1': 0, 'bleu2': 0, 'bleu3': 0, 'bleu4': 0, 'distinct': 0}        

        self.model = KGSFModel(self.opt, self.dict)
        if self.on_cuda:
            self.model.cuda()

        states = {}
        self.init_optim(
            [p for p in self.model.parameters() if p.requires_grad])

    def update_rec_accuracy(self, rec_loss, scores, labels):
        batchsize = len(labels.view(-1).tolist())
        result = scores.cpu()
        result = result[:, torch.LongTensor(self.movie_id)]
        _, top_items = torch.topk(result, k=50, dim=1)
        for i in range(batchsize):
            if labels[i].item() == 0:
                continue
            label = self.movie_id.index(labels[i].item())
            self.rec_accuracy['cnt'] += 1
            self.rec_accuracy['top1_acc'] += int(label in top_items[i][:1].tolist())
            self.rec_accuracy['top10_acc'] += int(label in top_items[i][:10].tolist())
            self.rec_accuracy['top50_acc'] += int(label in top_items[i][:50].tolist())
        self.rec_accuracy['loss_sum'] += rec_loss

    def update_res_accuracy(self, result, labels):
        def bleu(sentence, label):
            weights = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
            bleus = []
            for i in range(4):
                bleus.append(sentence_bleu([label], sentence, weights=weights[i]))
            return bleus

        def distinct(sentences):
            cnt, table = 0, set()
            for sentence in sentences:
                for i in range(len(sentence)-3):
                    vier = str(sentence[i]) + ' ' + str(sentence[i+1]) + ' ' + str(sentence[i+2]) + ' ' + str(sentence[i+3])
                    cnt += 1
                    table.add(vier)
            ret = len(table) / len(sentences)
            return ret

        tmp = []
        for res, labl in zip(result, labels):
            tmp.append(res)
            bleus = bleu(res, labl)
            self.res_accuracy['cnt'] += 1
            self.res_accuracy['bleu1'] += bleus[0]
            self.res_accuracy['bleu2'] += bleus[1]
            self.res_accuracy['bleu3'] += bleus[2]
            self.res_accuracy['bleu4'] += bleus[3]
        self.res_accuracy['distinct'] = distinct(tmp)

    def vector2sentence(self, sentences_idx):
        sentences=[]
        for sentence_idx in sentences_idx.numpy().tolist():
            sentence=[]
            for word in sentence_idx:
                if word > 3:
                    sentence.append(self.index2word[word])
                elif word==3:
                    sentence.append('_UNK_')
            sentences.append(sentence)
        return sentences
    
    def train_block(self, trainset, n_epoch, is_pre=False, best=0):
        if not self.is_finetune:
            losses, best_rec = [], best
            for i in range(n_epoch):
                print("Epoch "+str(i))
                print("initializing dataset, not fine-tuning")
                train_set = CRSdataset(trainset.data_process(), self.opt['n_dbpedia'], self.opt['n_concept'])
                train_dataloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.batchsize,
                                                               shuffle=False)
                cnt = 0
                print("loading, not fine-tuning")
                for context, c_lengths, response, r_length, mask_response, mask_r_length, entity, entity_vector, movie, concept_mask, dbpedia_mask, concept_vec, db_vec, rec in tqdm(
                        train_dataloader):
                    mentioned = []
                    batchsize = context.shape[0]
                    for b in range(batchsize):
                        tmp = entity[b].nonzero().view(-1).tolist()
                        mentioned.append(tmp)
                    self.model.train()
                    self.optimizer.zero_grad()

                    scores, preds, rec_scores, rec_loss, gen_loss, mask_loss, info_db_loss = self.model(concept_mask, dbpedia_mask, concept_vec, db_vec, entity_vector.cuda(), mentioned, movie, rec, context.cuda(), response.cuda(), mask_response.cuda(), test=False)

                    if is_pre:
                        cross_loss = info_db_loss
                        losses.append([info_db_loss])
                    else:
                        cross_loss = rec_loss + 0.025 * info_db_loss
                        losses.append([info_db_loss, rec_loss])

                    self.backward(cross_loss)
                    self.update_params()
                    if cnt % 50 == 0:
                        if not is_pre:
                            print('recommendation loss is %f' % (sum([l[1] for l in losses]) / len(losses)))
                        print('kg loss is %f' % (sum([l[0] for l in losses]) / len(losses)))
                        losses = []
                    cnt += 1

                if not is_pre:
                    rec_accuracy = self.val()

                    if best_rec > rec_accuracy["top50_acc"] + rec_accuracy["top1_acc"]:
                        break
                    else:
                        best_rec = rec_accuracy["top50_acc"] + rec_accuracy["top1_acc"]
                        self.model.save_model()
                        print("recommendation model saved once------------------------------------------------")
            return best_rec
        else:
            losses, best_res = [], best
            for i in range(n_epoch):
                print("Epoch "+str(i)+"/"+str(n_epoch))
                print("initializing dataset, fine-tuning")
                train_set = CRSdataset(trainset.data_process(True), self.opt['n_dbpedia'], self.opt['n_concept'])
                train_dataset_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.batchsize, shuffle=False)
                cnt = 0
                print("loading, fine-tuning")
                for context, c_lengths, response, r_length, mask_response, mask_r_length, entity, entity_vector, movie, concept_mask, dbpedia_mask, concept_vec, db_vec, rec in tqdm(
                        train_dataset_loader):
                    mentioned = []
                    batch_size = context.shape[0]
                    for b in range(batch_size):
                        tmp = entity[b].nonzero().view(-1).tolist()
                        mentioned.append(tmp)
                    self.model.train()
                    self.optimizer.zero_grad()

                    scores, preds, rec_scores, rec_loss, gen_loss, mask_loss, info_db_loss = self.model(
                        concept_mask, dbpedia_mask, concept_vec, db_vec, entity_vector.cuda(), mentioned, movie, rec, context.cuda(), response.cuda(), mask_response.cuda(), test=False)

                    cross_loss = gen_loss
                    losses.append([gen_loss])
                    self.backward(cross_loss)
                    self.update_params()
                    if cnt % 50 == 0:
                        print('response loss is %f' % (sum([l[0] for l in losses]) / len(losses)))
                        losses = []
                    cnt += 1

                output = self.val(True)
                if best_res >= output["distinct"]:
                    best_res = output["distinct"]
                    self.model.save_model()
                    print("generator model saved once------------------------------------------------")

            _ = self.val(is_test=True)
        
    
    def val(self, is_test=False, gen_conv=False):
        self.rec_accuracy = {"cnt": 0, "top1_acc": 0, "top10_acc": 0, "top50_acc": 0, "loss_sum": 0}
        self.res_accuracy={"cnt":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0}
        self.model.eval()
        if is_test:
            val_dataset = dataset('data/test_data.jsonl', self.opt)
        else:
            val_dataset = dataset('data/valid_data.jsonl', self.opt)
        if self.is_finetune:
            val_set = CRSdataset(val_dataset.data_process(True), self.opt['n_dbpedia'], self.opt['n_concept'])
        else:
            val_set=CRSdataset(val_dataset.data_process(),self.opt['n_dbpedia'],self.opt['n_concept'])
        val_dataloader = torch.utils.data.DataLoader(dataset=val_set,batch_size=self.batchsize,shuffle=False)
        
        if not self.is_finetune:
            recs = []
            for context, c_lengths, response, r_length, mask_response, mask_r_length, entity, entity_vector, movie, concept_mask, dbpedia_mask, concept_vec, db_vec, rec in tqdm(
                    val_dataloader):
                with torch.no_grad():
                    mentioned = []
                    batchsize = context.shape[0]
                    for b in range(batchsize):
                        tmp = entity[b].nonzero().view(-1).tolist()
                        mentioned.append(tmp)
                    scores, preds, rec_scores, rec_loss, _, mask_loss, info_db_loss = self.model(
                        concept_mask, dbpedia_mask, concept_vec, db_vec, entity_vector.cuda(), mentioned, movie, rec, context.cuda(), response.cuda(), mask_response.cuda(), test=True, maxlen=20, bsz=batchsize)

                recs.extend(rec.cpu())
                self.update_rec_accuracy(rec_loss, rec_scores, movie)

            output = {key: self.rec_accuracy[key] / self.rec_accuracy["cnt"] for key in ["top1_acc", "top10_acc", "top50_acc", "loss_sum"]}
            print(output)

            return output
        else:
            if gen_conv:
                f = open('generated_conversations.txt', 'w', encoding='utf-8')
            else:
                predictions, labels, contexts, losses, recs = [], [], [], [], []
            for context, c_lengths, response, r_length, mask_response, mask_r_length, entity, entity_vector, movie, concept_mask, dbpedia_mask, concept_vec, db_vec, rec in tqdm(
                    val_dataloader):
                with torch.no_grad():
                    mentioned = []
                    batchsize = context.shape[0]
                    for b in range(batchsize):
                        tmp = entity[b].nonzero().view(-1).tolist()
                        mentioned.append(tmp)
                    _, _, _, _, gen_loss, mask_loss, info_db_loss = self.model(
                        concept_mask, dbpedia_mask, concept_vec, db_vec, entity_vector.cuda(), mentioned, movie, rec, context.cuda(), response.cuda(), mask_response.cuda(), test=False)
                    scores, preds, rec_scores, rec_loss, _, mask_loss, info_db_loss = self.model(
                        concept_mask, dbpedia_mask, concept_vec, db_vec, entity_vector.cuda(), mentioned, movie, rec, context.cuda(), response.cuda(), mask_response.cuda(), test=True, maxlen=20, bsz=batchsize)
                    if gen_conv:
                        last_context = self.vector2sentence(context.cpu())
                        prediction = self.vector2sentence(preds.cpu())
                        for i in range(min(len(last_context), len(prediction))):
                            f.writelines('USR: ' + (' '.join(last_context[i])).split("_split_")[-1] + '\n')
                            f.writelines('RE:  ' + ' '.join(prediction[i]) + '\n')
                    else:
                        labels.extend(self.vector2sentence(response.cpu()))
                        predictions.extend(self.vector2sentence(preds.cpu()))
                        contexts.extend(self.vector2sentence(context.cpu()))
                        recs.extend(rec.cpu())
                        losses.append(torch.mean(gen_loss))
            if gen_conv:
                f.close()
                print("GENERATED")
            else:
                self.update_res_accuracy(predictions,labels)
                output = {key: self.res_accuracy[key] / self.res_accuracy["cnt"] for key in ["bleu1", "bleu2", "bleu3", "bleu4"]}
                output['distinct'] = self.res_accuracy['distinct']
                print(output)
                
                f = open('context_test.txt', 'w', encoding='utf-8')
                f.writelines([' '.join(sen) + '\n' for sen in contexts])
                f.close()

                f = open('output_test.txt', 'w', encoding='utf-8')
                f.writelines([' '.join(sen) + '\n' for sen in predictions])
                f.close()
                return output
    
    def train(self):
        print("start training")
        if not self.is_finetune:
            print("not fine-tuning")
            best = self.train_block(self.trainset, 3, is_pre=True)
            print("masked loss pre-trained")
            _ = self.train_block(self.trainset, self.epoch, is_pre=False, best=best)
            _ = self.val(is_test=True)
        else:
            print("fine tuning")
            self.model.load_model()
            self.train_block(self.trainset, self.epoch*3, best=1000)

    @classmethod
    def optim_opts(self):
        optims = {k.lower(): v for k, v in optim.__dict__.items()
                  if not k.startswith('__') and k[0].isupper()}
        return optims

    def init_optim(self, params):
        opt = self.opt

        lr = opt['learningrate']
        kwargs = {'lr': lr}
        kwargs['amsgrad'] = True
        kwargs['betas'] = (0.9, 0.999)

        optim_class = self.optim_opts()[opt['optimizer']]
        self.optimizer = optim_class(params, **kwargs)

    def backward(self, loss):
        loss.backward()

    def update_params(self):
        update_freq = 1
        if update_freq > 1:
            self._number_grad_accum = (self._number_grad_accum + 1) % update_freq
            if self._number_grad_accum != 0:
                return

        if self.opt['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.opt['gradient_clip']
            )

        self.optimizer.step()


if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-makedata","--makedata",type=bool,default=False)
    argParser.add_argument("-is_finetune","--is_finetune",type=bool,default=False)
    argParser.add_argument("-epoch","--epoch",type=int,default=30)
    argParser.add_argument("-batch_size","--batch_size",type=int,default=16)
    argParser.add_argument("-on_cuda", "--on_cuda", type=bool, default=True)
    argParser.add_argument("-learningrate", "--learningrate", type=float, default=1e-3)
    argParser.add_argument("-optimizer","--optimizer",type=str,default='adam')
    argParser.add_argument("-gradient_clip","--gradient_clip",type=float,default=0.1)
    argParser.add_argument("-num_bases", "--num_bases", type=int, default=8)
    argParser.add_argument("-dim", "--dim", type=int, default=128)
    argParser.add_argument("-n_dbpedia","--n_dbpedia",type=int,default=64368)
    argParser.add_argument("-n_concept","--n_concept",type=int,default=690988)
    argParser.add_argument("-embedding_size","--embedding_size",type=int,default=300)
    argParser.add_argument("-max_c_length","--max_c_length",type=int,default=256)
    argParser.add_argument("-max_r_length","--max_r_length",type=int,default=30)
    argParser.add_argument("-max_count","--max_count",type=int,default=5)
    argParser.add_argument("-n_heads","--n_heads",type=int,default=2)
    argParser.add_argument("-n_layers","--n_layers",type=int,default=2)
    argParser.add_argument("-ffn_size","--ffn_size",type=int,default=300)
    argParser.add_argument("-dropout","--dropout",type=float,default=0.1)
    argParser.add_argument("-learn_positional_embeddings","--learn_positional_embeddings",type=bool,default=False)
    argParser.add_argument("-extract_features","--extract_features",type=bool,default=False)
    argParser.add_argument("-generate_conversations","--generate_conversations",type=bool,default=False)
    args = argParser.parse_args()

    print(vars(args))

    if args.extract_features == True:
        kgsf_agent = TrainModel(vars(args))
        kgsf_agent.model.load_model()
        kgsf_agent.model.extract_features()
    elif args.generate_conversations == True:
        args.is_finetune = True
        kgsf_agent = TrainModel(vars(args))
        kgsf_agent.model.load_model()
        kgsf_agent.val(is_test=True, gen_conv=False)
    else:
        import time
        start_time = time.time()
        if args.is_finetune==False:
            kgsf_agent = TrainModel(vars(args))
            print("model initialized, not fine-tuning")
            kgsf_agent.train()
        else:
            kgsf_agent = TrainModel(vars(args))
            print("model initialized, fine-tuning")
            kgsf_agent.model.load_model()
            kgsf_agent.train()
        print("DONE\n")
        res = kgsf_agent.val(True)
        end_time = time.time()
        elapse = end_time-start_time
        print("Elapse: "+str(elapse/60))
