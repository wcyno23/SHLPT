import logging
import numpy as np
import torch
import os
import time
import GPUtil

from options.base_options import BaseOptions
from utils import utils
from approaches.common import ContinualLearning
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from fairscale.optim.grad_scaler import ShardedGradScaler

logger = logging.getLogger(__name__)


def train(algo: ContinualLearning):
    args = algo.args
    if args.distributed:
        logger.info("set up distributed model")

        # algo.pri_head = torch.nn.parallel.DistributedDataParallel(algo.pri_head, device_ids=[args.local_rank],
        #                                                           output_device=args.local_rank,
        #                                                           find_unused_parameters=False)
        algo.pri_head = ShardedDDP(algo.pri_head, algo.optimizer_pri)

    # initial the datasets
    logger.info("***** Running training *****")
    for k, v in algo.train_datasets.items():
        logger.info("Primary Task= {} Num examples = {}".format(k, len(v)))
        if args.is_save_log:
            algo.add_log("Primary Task= {} Num examples = {}".format(k, len(v)))

    logger.info("There are %d tasks to deal with iter %d an epoch" % (len(algo.task_list["train"]), algo.t_total))

    # training data flow
    total_iter = 0
    last_iter = 0
    downstream_iter = 0

    primary_table = np.zeros((algo.task_num + 1, len(args.train_tasks)))
    accuracy_table = np.zeros((algo.task_num + 1, len(args.train_tasks)))

    # logger.info(">>> eval before training...")
    # if args.is_save_log:
    #     algo.add_log(">>> eval before training...")

    #     for i in range(len(args.train_tasks)):
    #         algo.evaluate("Valid", args.train_tasks[i])
    #         _, test_acc, test_primary_metric_score = algo.evaluate("Test", args.train_tasks[i])
    #         primary_table[0, i] = test_primary_metric_score
    #         accuracy_table[0, i] = test_acc

    # logger.info('*' * 50)
    # logger.info('Primary Metric Score =')
    # for i in range(primary_table.shape[0]):
    #     primary_agg = ""
    #     for j in range(primary_table.shape[1]):
    #         primary_agg += '{:5.1f}% '.format(100 * primary_table[i, j])
    #     logger.info(primary_agg)
    #     if args.is_save_log:
    #         algo.add_log(primary_agg)
    # logger.info('Accuracies = ')
    # for i in range(accuracy_table.shape[0]):
    #     acc_agg = ""
    #     for j in range(accuracy_table.shape[1]):
    #         acc_agg += '{:5.1f}%'.format(100 * accuracy_table[i, j])
    #     logger.info(acc_agg)
    #     if args.is_save_log:
    #         algo.add_log(acc_agg)
    # logger.info('*' * 50)
    # if not eval before training
    algo.pri_head.train()
    algo.base_model.train()
    for stage_id in range(algo.task_num):
        step_in_stage = int(algo.task_list["train"][stage_id][0] * args.epoch)

        logger.info(">>> No stage = %d", stage_id)
        logger.info(">>> Num Warmup Steps = %d", int(args.warmup_frac * step_in_stage))
        logger.info(">>> Total optimization steps = {}. Will eval every {}".format(step_in_stage, args.eval_every))
        # do not use schedule in shlpt
        if args.algo not in ["progressive_prompts", "shlpt", "t5basic", "t5ewc", "l2p", "coda", 't5er', 't5derpp']:
            algo.setup_schedule(step_in_stage)

        if args.algo in ["shlpt", "l2p", "coda"]:
            algo.do_each_stage(stage_id=stage_id)
        else:
            algo.do_each_stage()

        stage_info = algo.task_generator.load_stage_dataset(stage_id)
        logger.info(">>> Load the dataset in %d-domian, the set contains following data   <<<<<<<" % (stage_id))
        for data_info in stage_info:
            logger.info(">>>        dataset name %s with %d batches             <<<<<<<" % (
                data_info["task_name"], data_info["datalen"]))

        early_stop_epoch = 0
        best_primary_metric_score = -1

        # begin_epoch = [0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        begin_epoch = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        stop_epoch = [150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150]
        if "short" in args.dataset_set:
            accum_epoch = 40
        elif "dis" in args.dataset_set:
            if args.accum_epoch > 25:
                accum_epoch = args.accum_epoch
            else:
                accum_epoch = 25
        elif "scl" in args.dataset_set:
            if args.accum_epoch > 25:
                accum_epoch = args.accum_epoch
            else:
                accum_epoch = 25
        else:
            accum_epoch = args.accum_epoch
        for epoch in range(int(args.epoch)):
            algo.task_generator.get_dataset_iter("train")
            losses = []
            cls_losses = []
            hsc_losses = []
            asc_losses = []
            total_steps = algo.task_list["train"][stage_id][0]
            # train the model
            for step in range(algo.task_list["train"][stage_id][0] // args.n_gpu):
                task_info, batch = algo.task_generator.get_batch_data(step)
                if batch is None:
                    break

                task_flag = False if task_info["task_type"] in ["mlm", "lm"] else True
                loss = algo.continual_train(batch, task_info, step, total_steps, cls_losses, hsc_losses, asc_losses)
                losses.append(loss.item())

                if (total_iter + 1) % args.loss_every == 0:
                    # if (total_iter) % args.gradient_accumulation_steps == 0:
                    logger.info(
                        "Iter %d, stage %d, epoch %d, step %d : average loss %f cls loss %f hsc loss %f asc loss %f." % (
                            total_iter, stage_id, epoch, step, np.average(losses), np.average(cls_losses),
                            np.average(hsc_losses), np.average(asc_losses)))
                    if args.is_save_log:
                        algo.add_log(
                            "Iter %d, stage %d, epoch %d, step %d : average loss %f cls loss %f hsc loss %f asc loss %f." % (
                                total_iter, stage_id, epoch, step, np.average(losses), np.average(cls_losses),
                                np.average(hsc_losses), np.average(asc_losses)))

                total_iter += 1
                if task_flag:
                    downstream_iter += 1
            if args.local_rank in [0, -1]:
                logger.info("stage %d, epoch %d: average loss %f . " % (stage_id, epoch, np.average(losses)))
                if args.is_save_log:
                    algo.add_log("stage %d, epoch %d: average loss %f . " % (stage_id, epoch, np.average(losses)))
            if epoch >= begin_epoch[stage_id] and args.local_rank in [0, -1] and (epoch + 1) % args.eval_every == 0:
                logger.info(">>> eval after train epoch %d..." % (epoch + 1))
                if args.is_save_log:
                    algo.add_log(">>> eval after train epoch %d..." % (epoch + 1))
                _, _, valid_primary_metric_score = algo.evaluate("Valid", "stage_%d" % stage_id)

                if valid_primary_metric_score > best_primary_metric_score:
                    early_stop_epoch = 0
                    best_primary_metric_score = valid_primary_metric_score
                    algo.save_best_models_to_cache()
                # algo.save_best_models_to_cache()
                algo.save_log()
                early_stop_epoch += 1
                if early_stop_epoch >= accum_epoch or epoch >= stop_epoch[stage_id]:  # 15
                    logger.info(">>> stop after train epoch %d..." % (epoch + 1))
                    break

        logger.info(">>> load the best model in stage %d..." % (stage_id))
        algo.load_best_models_from_cache()

        if args.algo in ["l2p", "shlpt"]:
            algo.do_after_stage(stage_id=stage_id)
        else:
            algo.do_after_stage()

        logger.info(">>> eval after train stage %d..." % (stage_id))
        if args.is_save_log:
            algo.add_log(">>> eval after train stage %d..." % (stage_id))

        if args.local_rank in [0, -1]:
            if args.algo == "progressive_prompts" and args.progressive:
                algo.evaluate("Valid", args.train_tasks[stage_id])
                _, test_acc, test_primary_metric_score = algo.evaluate("Test", args.train_tasks[stage_id])
                primary_table[stage_id + 1, stage_id] = test_primary_metric_score
                accuracy_table[stage_id + 1, stage_id] = test_acc
            elif args.algo == "l2p" and args.task_prompt:
                algo.evaluate("Valid", args.train_tasks[stage_id])
                _, test_acc, test_primary_metric_score = algo.evaluate("Test", args.train_tasks[stage_id])
                primary_table[stage_id + 1, stage_id] = test_primary_metric_score
                accuracy_table[stage_id + 1, stage_id] = test_acc
            elif args.algo == "coda":
                algo.evaluate("Valid", args.train_tasks[stage_id])
                _, test_acc, test_primary_metric_score = algo.evaluate("Test", args.train_tasks[stage_id])
                primary_table[stage_id + 1, stage_id] = test_primary_metric_score
                accuracy_table[stage_id + 1, stage_id] = test_acc
            else:
                for i in range(stage_id + 1):
                    # for i in range(len(args.train_tasks)):
                    # algo.evaluate("Train", args.train_tasks[i])
                    algo.evaluate("Valid", args.train_tasks[i])
                    _, test_acc, test_primary_metric_score = algo.evaluate("Test", args.train_tasks[i])
                    primary_table[stage_id + 1, i] = test_primary_metric_score
                    accuracy_table[stage_id + 1, i] = test_acc

        logger.info('*' * 50)
        logger.info('Primary Metric Score =')
        for i in range(primary_table.shape[0]):
            primary_metric_score_agg = ""
            for j in range(primary_table.shape[1]):
                primary_metric_score_agg += '{:5.1f}% '.format(100 * primary_table[i, j])
            logger.info(primary_metric_score_agg)
            if args.is_save_log:
                algo.add_log(primary_metric_score_agg)
        logger.info('Accuracies = ')
        for i in range(accuracy_table.shape[0]):
            acc_agg = ""
            for j in range(accuracy_table.shape[1]):
                acc_agg += '{:5.1f}%'.format(100 * accuracy_table[i, j])
            logger.info(acc_agg)
            if args.is_save_log:
                algo.add_log(acc_agg)
        logger.info('*' * 50)

        algo.save_log()
        if args.save_model:
            logger.info(">>> save best model after train stage %d..." % (stage_id))
            algo.save_models(stage_id)
    if args.save_model:
        logger.info(">>> save best model after train stage %d..." % (stage_id))
        algo.save_models(stage_id)


def main():
    args = BaseOptions().parse()
    if args.distributed:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        print(args.local_rank)
    if args.local_rank == -1:
        torch.cuda.set_device(args.gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if args.no_cuda:
        args.n_gpu = 0
    logger.info(">>> Run on the device %s, gpu num %d" % (args.device, args.n_gpu))
    # Set seed
    utils.set_seed(args)

    logger.info("training offline")
    saver = utils.offline_saver(args.model_name, "../save_model/", args.load_name)

    if args.algo == "shlpt":
        from approaches.cl_shlpt import SHLPTCL
        algos = SHLPTCL(args, saver)
    else:
        raise ValueError("There is no CL algo named %s !!!" % args.algo)
    algos.setup_base_models()
    algos.setup_datasets()
    algos.setup_head_models()
    algos.setup_tasks()
    algos.setup_scaler()

    algos.setup_external()
    # initial optimizer
    if args.froze_base:
        for param in algos.base_model.parameters():
            param.requires_grad = False

    algos.setup_optimizer()

    if args.local_rank != -1:
        torch.distributed.barrier()
    train(algos)


if __name__ == "__main__":
    main()

