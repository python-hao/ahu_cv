import torch,sys,os
sys.path.append(os.getcwd())
from torch import nn
from torch import optim
from pre_learning.xhaoTools import d2l,plot
import threading
class ReShape(nn.Module):
    def forward(self, x):
        return x

net = nn.Sequential(
    nn.Conv2d(1,96,kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96,256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256,384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384,384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384,256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096, 10)
)
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
net = net.to(device)

def train(p):

    # X = torch.randn(1,1,224,224)
    # for layer in net:
    #     X= layer(X)
    #     print(layer.__class__.__name__,X.shape, sep="\t")

    batch_size = 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

    lr = 0.01
    num_epochs = 10
    loss = nn.CrossEntropyLoss()
    updater = optim.Adam(net.parameters(), lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater,p)

if __name__ == '__main__':
    xplt = plot.AnimatorPlot(frames=10, interval=1)
    xplt.set_axes(xlabel='epoch', ylabel='rmse', legend=['train', 'valid'], xlim=(1, 10), ylim=(0, 5),
                  xscale='linear', yscale='linear')
    th = threading.Thread(target=train, args=(xplt,))
    th.start()
    xplt.show()