# MYYOLO

## TODO

- [x] 学习率的问题：使用`DataParallel`后loss计算用的是mean，学习率也没有缩放。另外原来用bs=12时，lr=2e-4，现在2快GPU用的是bs=32，应该相应缩放。（实际用lr=4e-4训练60epochs，发现较之前高了~0.35%）
- [x] route层的处理：现在实现route层的方式即使是只有1个层进行连接也会当成一个concat操作。这在QAT中留下了ffunc以及它的quant param，这是不必要的。
- [x] 支持更多的操作：要实现SE模块，`avgpool`和`scale_channels`是必要的。
- [ ] fuse时更加智能的处理：现在的fuse实现是只要是conv+bn+act/conv+bn/conv+act都可以。实际上act中只有ReLU是安全的。应该改为只对ReLU起效。另外train状态下的模型和eval下fuse表现不一样（指BN融合进conv），应该加入一个选项来控制。
- [ ] 支持更多的 backbone (目前来看 GhostNet > MobileNetV3 > MobileNetV2):
    + [x] MobileNetV2
    + [ ] MobileNetV3
    + [ ] GhostNet
    + [x] RegNetX
    + [ ] RegNetY
    + [ ] ShuffleNetV2
- [ ] 支持半精度训练（fp16）
- [ ] 修正 benchmark，支持仅测试前向推断过程（此过程应该不需要训练好的权重？）
- [ ] 实现 FPN 部分的 NAS 方法。
- [ ] 支持更多的数据集：
    + [x] VOC
    + [ ] COCO
    + [ ] DOTA
- [ ] 实现对任意 groups 卷积的剪枝方法（RegNet有此需求）