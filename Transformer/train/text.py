import torch.nn as nn
import collections
import os
import torch
from torchtext.vocab import vocab, GloVe
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, TensorDataset
from net import Text_Model

class Config(object):
    """配置参数"""
    def __init__(self):
        self.model_name = 'Transformer'
        self.embedding_pretrained = None  # 预训练词向量
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
        self.dropout = 0.5 # 随机失活
        self.num_classes = 2  # 类别数
        self.num_epochs = 20  # epoch数
        self.batch_size = 20  # mini-batch大小
        self.pad_size = 500   # 每句话处理成的长度(短填长切)
        self.n_vocab = None#这里需要读取数据的部分进行赋值
        self.learning_rate = 5e-4  # 学习率
        self.embed = 300  # 词向量维度
        self.dim_model = 300
        self.hidden = 1024
        self.last_hidden = 512
        self.num_head = 5
        self.num_encoder = 2
        self.checkpoint_path = './model.ckpt'

torch.manual_seed(1234)


class ImdbDataset(Dataset):
    def __init__(
            self, folder_path="./aclImdb", is_train=True, is_small=False
    ) -> None:
        super().__init__()
        self.data, self.labels = self.read_dataset(folder_path, is_train, is_small)

    # 读取数据
    def read_dataset(
            self,
            folder_path,
            is_train,
            small
    ):
        data, labels = [], []
        for label in ("pos", "neg"):
            folder_name = os.path.join(
                folder_path, "train" if is_train else "test", label
            )
            for file in tqdm(os.listdir(folder_name)):
                with open(os.path.join(folder_name, file), "rb") as f:
                    text = f.read().decode("utf-8").replace("\n", "").lower()
                    data.append(text)
                    labels.append(1 if label == "pos" else 0)
        # random.shuffle(data)
        # random.shuffle(labels)
        # 小样本训练，仅用于本机验证

        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], int(self.labels[index])

    def get_data(self):
        return self.data

    def get_labels(self):
        return self.labels


def get_tokenized(data):
    """获取数据集的词元列表"""

    def tokenizer(text):
        return [tok.lower() for tok in text.split(" ")]

    return [tokenizer(review) for review in data]


def get_vocab(data):
    """获取数据集的词汇表"""
    tokenized_data = get_tokenized(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    # 将min_freq设置为5，确保仅包括至少出现5次的单词
    vocab_freq = {"<UNK>": 0, "<PAD>": 1}
    # 添加满足词频条件的单词到词汇表，并分配索引
    for word, freq in counter.items():
        if freq >= 5:
            vocab_freq[word] = len(vocab_freq)

    # 构建词汇表对象并返回
    return vocab(vocab_freq)


def preprocess_imdb(train_data, vocab, config):
    """数据预处理，将数据转换成神经网络的输入形式"""
    max_l = config.pad_size  # 将每条评论通过截断或者补0，使得长度变成500

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [1] * (max_l - len(x))

    labels = train_data.get_labels()
    tokenized_data = get_tokenized(train_data.get_data())
    vocab_dict = vocab.get_stoi()
    features = torch.tensor(
        [pad([vocab_dict.get(word, 0) for word in words]) for words in tokenized_data]
    )
    labels = torch.tensor([label for label in labels])
    return features, labels


def load_data(config):
    """加载数据集"""
    train_data = ImdbDataset(folder_path="./aclImdb", is_train=True)
    test_data = ImdbDataset(folder_path="./aclImdb", is_train=False)
    print("输出第一句话：")
    print(train_data.__getitem__(1))
    vocab = get_vocab(train_data.get_data())
    train_set = TensorDataset(*preprocess_imdb(train_data, vocab, config))
    print("输出第一句话字典编码表示以及序列长度：")
    print(train_set.__getitem__(1), train_set.__getitem__(1)[0].shape)

    # 20%作为验证集
    # train_set, valid_set = torch.utils.data.random_split(
    #     train_set, [int(len(train_set) * 0.8), int(len(train_set) * 0.2)]
    # )
    test_set = TensorDataset(*preprocess_imdb(test_data, vocab, config))
    print(f"训练集大小{train_set.__len__()}")
    print(f"测试集大小{test_set.__len__()}")
    print(f"词表中单词个数:{len(vocab)}")
    train_iter = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    # valid_iter = DataLoader(valid_set, batch_size)
    test_iter = DataLoader(test_set, config.batch_size)
    return train_iter, test_iter, vocab

# train_data = ImdbDataset(is_train=True )
# test_data = ImdbDataset(is_train=False)
# vocab = get_vocab(train_data.get_data())
# print(f"词表中单词个数:{len(vocab)}")
# len_vocab=len(vocab)
# train_set = TensorDataset(*preprocess_imdb(train_data, vocab))
# test_set = TensorDataset(*preprocess_imdb(test_data, vocab))
# train_dataloader = DataLoader(
#     train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
# )
# test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE)
# load_data(config=Config())

# 预先定义配置
config = Config()
train_data, test_data, vocabs_size = load_data(config)  # 加载数据
config.n_vocab = len(vocabs_size) + 1  # 补充词表大小，词表一定要多留出来一个
model = Text_Model(config)  # 调用transformer的编码器
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
criterion = nn.CrossEntropyLoss()  # 多分类的任务
batch_size = config.batch_size

# 记录训练过程的数据
epoch_loss_values = []
metric_values = []
best_acc = 0.0
for epoch in range(config.num_epochs):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    # training
    model.train()
    for i, train_idx in enumerate(tqdm(train_data)):
        features, labels = train_idx
        features = features.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(features)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, train_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
        train_acc += (train_pred.detach() == labels.detach()).sum().item()
        train_loss += loss.item()
    model.eval()  # set the model to evaluation mode
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_data)):
            features, labels = batch
            features = features.cuda()
            labels = labels.cuda()
            outputs = model(features)

            loss = criterion(outputs, labels)

            _, val_pred = torch.max(outputs, 1)
            val_acc += (
                        val_pred.cpu() == labels.cpu()).sum().item()  # get the index of the class with the highest probability
            val_loss += loss.item()
    print(
        f'训练信息：[{epoch + 1:03d}/{config.num_epochs:03d}] Train Acc: {train_acc / 25000:3.5f} Loss: {train_loss / len(train_data):3.5f} | Val Acc: {val_acc / 25000:3.5f} loss: {val_loss / len(test_data):3.5f}')
    epoch_loss_values.append(train_loss / len(train_data))
    metric_values.append(val_acc / 25000)
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), config.checkpoint_path)
        print(f'saving model with acc {best_acc / 25000:.5f}')


