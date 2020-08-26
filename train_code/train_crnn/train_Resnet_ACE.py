from __future__ import print_function
import argparse
import random
import torch
# import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
# from torch.nn import CTCLoss
import utils2
import mydataset
import config
from online_test2 import val_model
from ACELoss import ACE
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
config.alphabet = config.alphabet_v2
config.nclass = len(config.alphabet) + 1
config.saved_model_prefix = 'resnet-ACE'
config.train_infofile = ['path_to_train_infofile1.txt','path_to_train_infofile2.txt']
config.val_infofile = 'path_to_test_infofile.txt'
config.keep_ratio = True
config.use_log = True
config.pretrained_model = 'resnet-ACE.pth'
config.batchSize = 128
config.workers = 24
config.adam = True

config.imgH = 30
config.imgW = 30
criterion = ACE()
config.lr = 0.0003
import os
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
log_filename = os.path.join('log/','loss_acc-'+config.saved_model_prefix + '.log')
if not os.path.exists('debug_files'):
    os.mkdir('debug_files')
if not os.path.exists(config.saved_model_dir):
    os.mkdir(config.saved_model_dir)
if config.use_log and not os.path.exists('log'):
    os.mkdir('log')
if config.use_log and os.path.exists(log_filename):
    os.remove(log_filename)
if config.experiment is None:
    config.experiment = 'expr'
if not os.path.exists(config.experiment):
    os.mkdir(config.experiment)

config.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", config.manualSeed)
random.seed(config.manualSeed)
np.random.seed(config.manualSeed)
torch.manual_seed(config.manualSeed)

# cudnn.benchmark = True
train_dataset = mydataset.MyDataset(info_filename=config.train_infofile)
assert train_dataset
if not config.random_sample:
    sampler = mydataset.randomSequentialSampler(train_dataset, config.batchSize)
else:
    sampler = None
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config.batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(config.workers),
    collate_fn=mydataset.alignCollate(imgH=config.imgH, imgW=config.imgW, keep_ratio=config.keep_ratio))

test_dataset = mydataset.MyDataset(
    info_filename=config.val_infofile, transform=mydataset.resizeNormalize((config.imgW, config.imgH), is_test=True))

converter = utils2.strLabelConverter(config.alphabet)
# criterion = CTCLoss(reduction='sum',zero_infinity=True)
best_acc = 0.5

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.conv = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.bn = nn.BatchNorm2d(64)
        self.cnn = nn.Sequential(*list(resnet.children())[4:-2])
        self.out = nn.Linear(512, config.nclass)



    def forward(self, input):
        input = F.relu(self.bn(self.conv(input)), True)
        input = F.max_pool2d(input, kernel_size=(2, 2), stride=(2, 2))
        input = self.cnn(input)
        input = input.permute(0, 2, 3, 1)
        input = F.softmax(self.out(input), dim=-1)
        return  input

resnet = Resnet()

# if config.pretrained_model!='' and os.path.exists(config.pretrained_model):
#     print('loading pretrained model from %s' % config.pretrained_model)
#     resnet.load_state_dict(torch.load(config.saved_model_dir+'/'+config.pretrained_model))
# else:
resnet.apply(weights_init)

# image = torch.FloatTensor(config.batchSize, 3, config.imgH, config.imgH)
# text = torch.IntTensor(config.batchSize * 5)
# length = torch.IntTensor(config.batchSize)
device = torch.device('cpu')
if config.cuda:
    resnet.cuda()
    # crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    # image = image.cuda()
    device = torch.device('cuda:0')
    # criterion = criterion.cuda()

# image = Variable(image)
# text = Variable(text)
# length = Variable(length)

# loss averager
loss_avg = utils2.averager()

# setup optimizer
if config.adam:
    optimizer = optim.Adam(resnet.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
elif config.adadelta:
    optimizer = optim.Adadelta(resnet.parameters(), lr=config.lr)
else:
    optimizer = optim.RMSprop(resnet.parameters(), lr=config.lr)

def val(net, dataset, criterion, max_iter=100):
    print('Start val')
    # for p in net.parameters():
    #     p.requires_grad = False

    num_correct,  num_all = val_model(config.val_infofile,net,True,log_file='compare-'+config.saved_model_prefix+'.log')
    accuracy = num_correct / num_all

    print('ocr_acc: %f' % (accuracy))
    if config.use_log:
        with open(log_filename, 'a') as f:
            f.write('ocr_acc:{}\n'.format(accuracy))
    global best_acc
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(resnet.state_dict(), '{}/{}_{}_{}.pth'.format(config.saved_model_dir, config.saved_model_prefix, epoch,
                                                               int(best_acc * 1000)))
    torch.save(resnet.state_dict(), '{}/{}.pth'.format(config.saved_model_dir, config.saved_model_prefix))


def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    image = cpu_images.to(device)

    label = converter.encode(cpu_texts)
    label = label.to(device)
    # utils.loadData(text, t)
    # utils.loadData(length, l)

    preds = net(image)  # seqLength x batchSize x alphabet_size
    # preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))  # seqLength x batchSize
    cost = criterion(preds, label) / batch_size
    if torch.isnan(cost):
        print(batch_size,cpu_texts)
    else:
        net.zero_grad()
        cost.backward()
        optimizer.step()
    return cost


for epoch in range(100):
    loss_avg.reset()
    print('epoch {}....'.format(epoch))
    train_iter = iter(train_loader)
    i = 0
    n_batch = len(train_loader)
    while i < len(train_loader):
        cost = trainBatch(resnet, criterion, optimizer)
        print('epoch: {} iter: {}/{} Train loss: {:.10f}'.format(epoch, i, n_batch, cost.item()))
        loss_avg.add(cost)
        loss_avg.add(cost)
        i += 1
    if config.use_log:
        with open(log_filename, 'a') as f:
            f.write('{}\n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')))
            f.write('train_loss:{}\n'.format(loss_avg.val()))

    if (epoch % 5 == 0 and epoch!=0):
        val(resnet, test_dataset, criterion)


