import torch
import torch.nn as nn


class LSTNet(nn.Module):
    def __init__(self, args, input_dim, output_dim):
        super(LSTNet, self).__init__()
        self.use_cuda = args.cuda
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')

        self.batch_size = args.batch_size
        self.P = args.window
        self.skip = args.skip
        self.hw = args.highway_window

        self.m_x = input_dim  # 7
        self.m_y = output_dim  # 1

        self.hidR = args.hidRNN
        self.hidC = args.hidCNN
        self.hidS = args.hidSkip
        self.Ck = args.CNN_kernel
        self.pt = (self.P - self.Ck) // self.skip

        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m_x)).to(self.device)
        self.GRU1 = nn.GRU(self.hidC, self.hidR).to(self.device)
        self.dropout = nn.Dropout(p=args.dropout).to(self.device)

        if self.skip > 0:
            self.GRUskip = nn.GRU(self.hidC, self.hidS).to(self.device)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m_y).to(self.device)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m_y).to(self.device)

        if self.hw > 0:
            self.highway = nn.Linear(self.hw * self.m_x, self.m_y).to(self.device)

        self.output = None
        if args.output_fun == 'sigmoid':
            self.output = torch.sigmoid
        if args.output_fun == 'tanh':
            self.output = torch.tanh

    def forward(self, x):
        batch_size = x.size(0)

        # CNN
        c = x.view(-1, 1, self.P, self.m_x)
        c = torch.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))

        # skip-rnn
        if self.skip > 0:
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)

        res = self.linear1(r)

        # highway
        if self.hw > 0:
            z = x[:, -self.hw:, :]
            z = z.view(-1, self.hw * self.m_x)
            z = self.highway(z)
            z = z.view(-1, self.m_y)
            res = res + z

        if self.output:
            res = self.output(res)
        return res
