
import torchtext
import torch
dataType = 'val2017'
datadir = '.'
import torchvision.transforms as transforms
import torchvision.models as models
import json
from rnndecoder import CNNEncoder, RNNDecoder

# from pycocotools.coco import COCO
# annFile = 'captions_val2017.json'
# root = './val2017'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
field = torchtext.data.Field(sequential=True,use_vocab=True, init_token= None, eos_token= '<eos>', tokenize='spacy', is_target=False, lower = True , batch_first = True)
inputfield = torchtext.data.Field(use_vocab=False, sequential= False, dtype = torch.float32)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
# with open(annFile, 'r') as fp:
#     annotations = json.load(fp)
#
# coco = COCO(annFile)
# keys = list(sorted(coco.imgs.keys()))
# trans = transforms.Compose([transforms.Resize(286), transforms.RandomResizedCrop(224),transforms.ToTensor(), normalize])
fields = {'input': inputfield,'target': field}
# vgg16 = models.vgg16(pretrained=True)
# vgg16.eval()
examples = []
# with torch.no_grad():
#     for img_id in keys:
#         ann_ids = coco.getAnnIds(imgIds=img_id)
#         anns = coco.loadAnns(ann_ids)
#         # target = [ann['caption'] for ann in anns]
#
#         path = coco.loadImgs(img_id)[0]['file_name']
#
#         img = Image.open(os.path.join(root, path)).convert('RGB')
#         img = trans(img)
#         img = img.unsqueeze_(0)
#         out = vgg16.features(img).view(-1).numpy()
#         for ann in anns:
#             target = ann['caption']
#             ex = {'input': out , 'target': target}
#             ex_fields = {k: [(k, v)] for k, v in fields.items()}
#             example = Example.fromdict(ex, ex_fields)
#             examples.append(example)



import pickle
# with open('data.pkl', 'wb') as fp:
#     pickle.dump(examples, fp)

with open('data.pkl', 'rb') as fp:
    examples = pickle.load(fp)

dataset = torchtext.data.Dataset(examples, fields)
field.build_vocab(dataset)
vocab = field.vocab

vocab_size = len(vocab)


from torchtext.data import Iterator, BucketIterator

train_ds, valid_ds , test_ds = dataset.split(split_ratio=[0.7,0.2,0.1],stratified=False)
print(len(valid_ds))
train_iter = Iterator(train_ds, batch_size=64, device=device, sort=False, sort_key = lambda x: len(x.target), repeat=False, shuffle=True)
valid_iter = Iterator(valid_ds, batch_size=64, device =device, sort= False, repeat= False)

epochs = 100
embedding_dim = 300
hidden_dim = 512
learning_rate = 0.01
batch_size = 64
input_dim = 512 #49*512
encoder = CNNEncoder(input_dim = input_dim, embedding_dim = embedding_dim)
decoder = RNNDecoder(hidden_dim,embedding_dim, vocab_size)
enc_optimizer = torch.optim.SGD(encoder.parameters(), learning_rate, momentum=0.9)
dec_optimizer = torch.optim.SGD(decoder.parameters(), learning_rate, momentum=0.9)
criterion = torch.nn.NLLLoss(ignore_index=vocab.stoi['<pad>'])

encoder.to(device)
decoder.to(device)

encoder.train()
decoder.train()
for i in train_iter:

    enc_optimizer.zero_grad()
    dec_optimizer.zero_grad()


    input = i.input.to(device)
    output = i.target.to(device)
    input = input.reshape(batch_size, -1, hidden_dim)
    features = encoder(input)
    x = torch.tensor([vocab.stoi['<bos>']]*batch_size, dtype= torch.long).unsqueeze_(1)
    hidden = decoder.reset_state(batch_size)
    total_loss = 0
    for i in range(output.shape[1]):
        predictions, hidden, _ = decoder(x, features, hidden)
        loss = criterion(predictions,output[:,i])
        total_loss += loss


    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.5)
    dec_optimizer.step()
    enc_optimizer.step()
    print(total_loss.item())

