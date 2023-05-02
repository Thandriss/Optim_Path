import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from core.utils import dist_util
from .inference import eval_dataset_ships
from core.data import make_data_loader_ships
from torch.utils.tensorboard import SummaryWriter


def do_eval(cfg, model, distributed, **kwargs):
    torch.cuda.empty_cache()

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    data_loader = make_data_loader_ships(cfg, False)
    # model.eval()
    device = torch.device(cfg.MODEL.DEVICE)
    result_dict = eval_dataset_ships(cfg, model, data_loader, device, 'pytorch')

    torch.cuda.empty_cache()
    return result_dict


def do_train(cfg,
             model,
             data_loader,
             optimizer,
             scheduler,
             checkpointer,
             device,
             arguments,
             args):

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    logger = logging.getLogger("CORE")
    logger.info("Start training ...")

    # Set model to train mode
    model.train()

    # Create tensorboard writer
    save_to_disk = dist_util.is_main_process()
    if args.use_tensorboard and save_to_disk:
        summary_writer = SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, 'tf_logs'))
    else:
        summary_writer = None

    # Prepare to train
    iters_per_epoch = len(data_loader)
    total_steps = iters_per_epoch * cfg.SOLVER.MAX_ITER
    start_epoch = arguments["epoch"]
    logger.info("Iterations per epoch: {0}. Total steps: {1}. Start epoch: {2}".format(iters_per_epoch, total_steps, start_epoch))

    # Epoch loop
    for epoch in range(start_epoch, cfg.SOLVER.MAX_ITER):
        arguments["epoch"] = epoch + 1

        # Create progress bar
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'lr', 'loss', 'loss_cls', 'loss_box', 'loss_obj', 'loss_rpn'))
        pbar = enumerate(data_loader)
        pbar = tqdm(pbar, total=len(data_loader))

        # Iteration loop
        loss_sum = 0.0
        loss_classifier = 0.0
        loss_box_reg = 0.0
        loss_objectness = 0.0
        loss_rpn_box_reg = 0.0

        for iteration, data_entry in pbar:
            global_step = epoch * iters_per_epoch + iteration

            images, rects, masks, rects_real_num = data_entry

            # Prepare images
            images = list(image.to(device) for image in images)

            # Prepare targets
            targets = []
            for i in range(len(images)):
                d = {}
                if rects_real_num[i] == 0:
                    d['boxes'] = torch.empty((0, 4), dtype=torch.float32).to(device)
                    d['labels'] = torch.empty((0,), dtype=torch.int64).to(device)
                else:
                    real_rects = rects[i][0:rects_real_num[i]]
                    d['boxes'] = real_rects.to(device)
                    d['labels'] = torch.ones(len(real_rects), dtype=torch.int64).to(device)
                targets.append(d)

            # Do prediction
            loss_dict = model.model(images, targets)

            # Calculate loss
            loss = sum(loss for loss in loss_dict.values())
            loss_sum += loss.item()
            loss_classifier += loss_dict["loss_classifier"].item()
            loss_box_reg += loss_dict["loss_box_reg"].item()
            loss_objectness += loss_dict["loss_objectness"].item()
            loss_rpn_box_reg += loss_dict["loss_rpn_box_reg"].item()

            # Do optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # Update progress bar
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0) # (GB)
            s = ('%10s' * 2 + '%10.4g' * 6) % (
                                                '%g/%g' % (epoch, cfg.SOLVER.MAX_ITER - 1),
                                                mem,
                                                optimizer.param_groups[0]['lr'],
                                                loss_sum / (iteration + 1),
                                                loss_classifier / (iteration + 1),
                                                loss_box_reg / (iteration + 1),
                                                loss_objectness / (iteration + 1),
                                                loss_rpn_box_reg / (iteration + 1))

            pbar.set_description(s)

        # Do evaluation
        if args.eval_step > 0 and epoch % args.eval_step == 0:
            print('\nEvaluation ...')
            res_dict = do_eval(cfg, model, distributed=args.distributed, iteration=global_step)
            print(('\n' + 'Evaluation results:' + '%10s' * 5) % ('loss', 'loss_cls', 'loss_box', 'loss_obj', 'loss_rpn'))
            print('                   ' + '%10.4g%10.4g%10.4g%10.4g%10.4g' % (res_dict['loss_sum'], res_dict['loss_classifier'], res_dict['loss_box_reg'], res_dict['loss_objectness'], res_dict['loss_rpn_box_reg']))

            if summary_writer:
                summary_writer.add_scalar('validation_losses/loss', res_dict['loss_sum'], global_step=global_step)
                summary_writer.add_scalar('validation_losses/loss_classifier', res_dict['loss_classifier'], global_step=global_step)
                summary_writer.add_scalar('validation_losses/loss_box_reg', res_dict['loss_box_reg'], global_step=global_step)
                summary_writer.add_scalar('validation_losses/loss_objectness', res_dict['loss_objectness'], global_step=global_step)
                summary_writer.add_scalar('validation_losses/loss_rpn_box_reg', res_dict['loss_rpn_box_reg'], global_step=global_step)
                summary_writer.flush()

            model.train()

        # Save epoch results
        if epoch % args.save_step == 0:
            checkpointer.save("model_{:06d}".format(global_step), **arguments)

            if summary_writer:
                with torch.no_grad():
                    summary_writer.add_scalar('losses/loss', loss_sum / (iteration + 1), global_step=global_step)
                    summary_writer.add_scalar('losses/loss_classifier', loss_classifier / (iteration + 1), global_step=global_step)
                    summary_writer.add_scalar('losses/loss_box_reg', loss_box_reg / (iteration + 1), global_step=global_step)
                    summary_writer.add_scalar('losses/loss_objectness', loss_objectness / (iteration + 1), global_step=global_step)
                    summary_writer.add_scalar('losses/loss_rpn_box_reg', loss_rpn_box_reg / (iteration + 1), global_step=global_step)
                    summary_writer.flush()

    # Save final model
    checkpointer.save("model_final", **arguments)

    return model