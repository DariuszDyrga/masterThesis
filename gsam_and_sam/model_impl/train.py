import argparse
import torch

from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from coronary import Coronary
from utility.log import Log
from utility.initialize import initialize

import sys; sys.path.append("..")
from gsam import GSAM, CosineScheduler, ProportionScheduler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=25, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=2.0, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--rho_max", default=2.0, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--rho_min", default=2.0, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--alpha", default=0.4, type=int, help="Rho parameter for SAM.")

    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = Coronary(args.batch_size, args.threads)

    log = Log(log_each=10)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)

    base_optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = CosineScheduler(T_max=args.epochs * len(dataset.train), max_value=args.learning_rate, min_value=0.0,
                                optimizer=base_optimizer)
    rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, max_lr=args.learning_rate, min_lr=0.0,
                                        max_value=args.rho_max, min_value=args.rho_min)

    optimizer = GSAM(params=model.parameters(), base_optimizer=base_optimizer, model=model, gsam_alpha=args.alpha,
                     rho_scheduler=rho_scheduler, adaptive=args.adaptive)

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)
            inputs = inputs.permute(0, 3, 2, 1)


            def loss_fn(predictions, targets):
                return smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing).mean()


            optimizer.set_closure(loss_fn, inputs, targets)
            predictions, loss = optimizer.step()

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler.step()
                optimizer.update_rho_t()

        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)
                inputs = inputs.permute(0, 3, 2, 1)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())

    log.flush()