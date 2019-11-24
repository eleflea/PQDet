import logging
import math
import os

import numpy as np
import torch
import torch.optim as optim

import config as cfg
from dataset.dataset import YOLODataset
from eval.evaluate import Evaluator
from model.yolov3 import YOLOv3


def adjust_lr(optimizer, steps):
    warmup_steps = cfg.WARMUP_EPOCHS * cfg.STEPS_PER_EPOCH
    max_steps = cfg.MAX_EPOCHS * cfg.STEPS_PER_EPOCH
    if steps < warmup_steps:
        lr = steps / warmup_steps * cfg.LEARN_RATE_INIT
    else:
        lr = cfg.LEARN_RATE_END + 0.5*(cfg.LEARN_RATE_INIT-cfg.LEARN_RATE_END) *\
            (1 + math.cos((steps-warmup_steps)/(max_steps-warmup_steps)*math.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def main():
    os.makedirs(cfg.WEIGHTS_DIR, exist_ok=True)

    dataset = YOLODataset()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True,
    )
    cfg.STEPS_PER_EPOCH = len(dataloader)
    # dataiterator = iter(dataloader)

    model = YOLOv3().cuda()
    # print(model)
    # model = torch.nn.DataParallel(model, device_ids=[0,1]) # multi-GPU
    model.init_weights()

    evaluator = Evaluator(model)

    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARN_RATE_INIT)

    global_step = 0
    init_eopch = 0

    if cfg.RESUME:
        state_dict = torch.load(cfg.RESUME_WEIGHTS)
        global_step = state_dict['step']
        model.load_state_dict(state_dict['model'])
        init_eopch = global_step // cfg.STEPS_PER_EPOCH
        print('resume at %d steps from %s' % (global_step, cfg.RESUME_WEIGHTS))

    print_loss_step = cfg.STEPS_PER_EPOCH // 10
    acc_loss = 0
    acc_cl = 0
    mAP = None
    model.train()
    for epoch in range(init_eopch, cfg.MAX_EPOCHS):
        for data in dataloader:
            global_step += 1
            image, label_sbbox, label_mbbox, label_lbbox,\
                sbbox, mbbox, lbbox = [item.cuda().squeeze(0) for item in data]
            
            lr = adjust_lr(optimizer, global_step)
            optimizer.zero_grad()
            loss, cl = model(image, (label_sbbox, label_mbbox, label_lbbox, sbbox, mbbox, lbbox))
            loss.backward()
            optimizer.step()
            acc_loss += loss.item()
            acc_cl += cl.item()

            if global_step % print_loss_step == 0:
                train_loss = acc_loss / print_loss_step
                train_closs = acc_cl / print_loss_step
                acc_loss = 0
                acc_cl = 0
                print('lr: %.6f\tepoch: %d/%d\tstep: %d\ttrain_loss: %.4f(%.4f)' %
                    (lr, epoch, cfg.MAX_EPOCHS, global_step, train_loss, train_closs))

        if epoch > 20:
            model.eval()
            APs = evaluator.mAP()
            for cls in APs:
                AP_mess = 'AP for %s = %.4f\n' % (cls, APs[cls])
                print(AP_mess.strip())
            mAP = np.mean([APs[cls] for cls in APs])
            mAP_mess = 'mAP = %.4f\n' % mAP
            print(mAP_mess.strip())
            model.train()
        model_name = f'model-{epoch}.pt' if mAP is None else f'model-{epoch}-{mAP:.4f}.pt'
        torch.save({'step': global_step, 'model': model.state_dict()},
            os.path.join(cfg.WEIGHTS_DIR, model_name))
        mAP = None
    
if __name__ == "__main__":
    main()
