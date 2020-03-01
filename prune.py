import torch
from pruning.slimming import SlimmingPruner
from model.yolov3 import YOLOv3
from thop import clever_format, profile

sp = SlimmingPruner(YOLOv3)
sp.prune()
model = sp.model
new_model = sp.new_model
inputs = torch.randn(1, 3, 512, 512).cuda()
flops, params = profile(model, inputs=(inputs, ), verbose=False)
flops, params = clever_format([flops, params], "%.3f")
flopsnew, paramsnew = profile(new_model, inputs=(inputs, ), verbose=False)
flopsnew, paramsnew = clever_format([flopsnew, paramsnew], "%.3f")
print("flops:{}->{}, params: {}->{}".format(flops, flopsnew, params, paramsnew))
sp.test()
