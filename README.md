# MYYOLO

## TODO

- 学习率的问题：使用`DataParallel`后loss计算用的是mean，学习率也没有缩放。另外原来用bs=12时，lr=2e-4，现在用的是bs=16，应该相应缩放。
- route层的处理：现在实现route层的方式即使是只有1个层进行连接也会当成一个concat操作。这在QAT中留下了ffunc以及它的quant param，这是不必要的。
- 支持更多的操作：要实现SE模块，`avgpool`和`scale_channels`是必要的。
- fuse时更加智能的处理：现在的fuse实现是只要是conv+bn+act/conv+bn/conv+act都可以。实际上act中只有ReLU是安全的。应该改为只对ReLU起效。另外train状态下的模型和eval下fuse表现不一样（指BN融合进conv），应该加入一个选项来控制。
