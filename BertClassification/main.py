from pytorch_transformers import BertTokenizer, AdamW
from pytorch_pretrained_bert import BertAdam
import argparse
from utils import build_dataset, DatasetIterator
from model import ClassifyModel
from train import train

parser = argparse.ArgumentParser()

# path arguments
parser.add_argument('--bert_path', type=str, default='D:/PythonProjects/语言模型/bert-base-chinese')
parser.add_argument('--train_path', type=str, default='data/train.txt')
parser.add_argument('--dev_path', type=str, default='data/dev.txt')
parser.add_argument('--test_path', type=str, default='data/test.txt')

# hyper parameters
parser.add_argument('--pad_length', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_class', type=int, default=10)
parser.add_argument('--hidden_size', type=int, default=768)
parser.add_argument('--require_improve', type=int, default=1000)
parser.add_argument('--EPOCH', type=int, default=10)
parser.add_argument('--dev_steps', type=int, default=100)


# other setting
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args()

if args.train:
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    print(tokenizer)
    train_data, dev_data, test_data = build_dataset(train_path=args.train_path,
                                                    dev_path=args.dev_path,
                                                    test_path=args.test_path,
                                                    tokenizer=tokenizer,
                                                    padding_size=args.pad_length
                                                    )
    train_iter = DatasetIterator(batches=train_data,
                                 batch_size=args.batch_size,
                                 device=args.device)
    dev_iter = DatasetIterator(batches=dev_data,
                                 batch_size=args.batch_size,
                                 device=args.device)
    test_iter = DatasetIterator(batches=test_data,
                                 batch_size=args.batch_size,
                                 device=args.device)
    model = ClassifyModel(bert_path=args.bert_path,
                          hidden_size=args.hidden_size,
                          num_class=args.num_class).to(args.device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=0.05, t_total=len(train_iter) * args.EPOCH)
    train(model=model,
          train_iter=train_iter,
          dev_iter=dev_iter,
          optimizer=optimizer,
          dev_steps=args.dev_steps,
          num_epochs=args.EPOCH)

