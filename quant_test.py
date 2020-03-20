import torch
from torch import nn, optim
from torch.quantization import QuantStub, DeQuantStub
from copy import deepcopy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Add(nn.Module):

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
        self.ffunc = nn.quantized.FloatFunctional()

    def forward(self, x):
        return self.ffunc.add(self.x(x), self.y(x))

# model = nn.Sequential(
#     QuantStub(),
#     nn.Conv2d(3, 16, 1, bias=False),
#     Add(
#         nn.Sequential(),
#         nn.Sequential(
#             nn.Conv2d(16, 16, 1, bias=False),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#         ),
#     ),
#     nn.Conv2d(16, 10, 3, stride=2, padding=1, bias=False),
#     nn.BatchNorm2d(10),
#     nn.AvgPool2d(14),
#     nn.Sigmoid(),
#     DeQuantStub(),
# )

# torch.quantization.fuse_modules(model, ['3', '4'], inplace=True)
# torch.quantization.fuse_modules(model[2].y, ['0', '1', '2'], inplace=True)
model = nn.Sequential(
    QuantStub(),
    nn.Conv2d(3, 16, 1, bias=False),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.Conv2d(16, 10, 3, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(10),
    nn.AvgPool2d(14),
    nn.Sigmoid(),
    DeQuantStub(),
)

torch.quantization.fuse_modules(model, [['1', '2', '3'], ['4', '5']], inplace=True)

model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(model, inplace=True)
optimizer = optim.Adam(model.parameters(), lr=1)
model = nn.DataParallel(model)
model.to(device)
print(model)

criterion = nn.BCELoss()

for epoch in range(10):
    model.train()

    inputs = torch.rand(2, 3, 28, 28)
    labels = torch.FloatTensor([[1,1,1,1,1,0,0,0,0,0], [1,1,1,1,1,0,0,0,0,0]])

    inputs = inputs.to(device)
    labels = labels.to(device)
    loss = criterion(model(inputs).view(2, 10), labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 2:
        model.apply(torch.quantization.disable_observer)
    if epoch >= 3:
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    quant_model = deepcopy(model.module)
    quant_model = torch.quantization.convert(quant_model.eval().cpu(), inplace=False)
    with torch.no_grad():
        out = quant_model(torch.rand(1, 3, 28, 28))
        print(out.view(10).tolist())
