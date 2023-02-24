# Adapted from:
# https://github.com/pytorch/examples/blob/9aad148615b7519eadfa1a60356116a50561f192/mnist/main.py

# Changes to the code are kept to a minimum to facilitate the comparison with the original example

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from bayestorch.distributions import get_mixture_log_scale_normal, get_softplus_inv_scale_normal
from bayestorch.nn import VariationalPosteriorModule


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        #self.dropout1 = nn.Dropout(0.25)
        #self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch, num_train_mc_samples, kl_div_weight):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        #output = model(data)
        #loss = F.nll_loss(output, target)
        output, kl_div = model(data, num_mc_samples=num_train_mc_samples, return_kl_div=True)
        loss = F.nll_loss(output, target, reduction="sum") + kl_div_weight * kl_div
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, num_test_mc_samples, kl_div_weight):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #output = model(data)
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            output, kl_div = model(data, num_mc_samples=num_test_mc_samples, return_kl_div=True)
            test_loss += (F.nll_loss(output, target, reduction="sum") + kl_div_weight * kl_div).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--normal-mixture-prior-weight', type=float, default=0.75,
                        help='mixture weight of normal mixture prior (default: 0.75)')
    parser.add_argument('--normal-mixture-prior-log-scale1', type=float, default=-1.0,
                        help='log scale of first component of normal mixture prior (default: -1.0)')
    parser.add_argument('--normal-mixture-prior-log-scale2', type=float, default=-6.0,
                        help='log scale of second component of normal mixture prior (default: -6.0)')
    parser.add_argument('--normal-posterior-softplus-inv-scale', type=float, default=-7.0,
                        help='inverse softplus scale of normal posterior (default: -7.0)')
    parser.add_argument('--kl-div-weight', type=float, default=1e-6,
                        help='Kullback-Leibler divergence weight (default: 1e-6)')
    parser.add_argument('--num-train-mc-samples', type=int, default=10,
                        help='number of Monte Carlo samples for training (default: 10)')
    parser.add_argument('--num-test-mc-samples', type=int, default=10,
                        help='number of Monte Carlo samples for testing (default: 10)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    #use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    #elif use_mps:
    #    device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net()

    # Prior and posterior are defined as in https://arxiv.org/abs/1505.05424
    # Prior arguments (WITHOUT gradient tracking)
    prior_builder, prior_kwargs = get_mixture_log_scale_normal(
        model.parameters(),
        weights=[args.normal_mixture_prior_weight, 1 - args.normal_mixture_prior_weight],
        locs=(0.0, 0.0),
        log_scales=(args.normal_mixture_prior_log_scale1, args.normal_mixture_prior_log_scale2)
    )

    # Posterior arguments (WITH gradient tracking)
    posterior_builder, posterior_kwargs = get_softplus_inv_scale_normal(
        model.parameters(), loc=0.0, softplus_inv_scale=args.normal_posterior_softplus_inv_scale, requires_grad=True,
    )

    # Bayesian model
    model = VariationalPosteriorModule(
        model, prior_builder, prior_kwargs, posterior_builder, posterior_kwargs
    ).to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, args.num_train_mc_samples, args.kl_div_weight)
        test(model, device, test_loader, args.num_test_mc_samples, args.kl_div_weight)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
