#!/usr/bin/env python
# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""execution script."""

import argparse
import os, warnings
import time

import torch

import utils.exp_utils as utils
from evaluator import Evaluator
from predictor import Predictor
from plotting import plot_batch_prediction

import numpy as np

import platform
import mlflow

for msg in ["Attempting to set identical bottom==top results",
            "This figure includes Axes that are not compatible with tight_layout",
            "Data has no positive values, and therefore cannot be log-scaled.",
            ".*invalid value encountered in double_scalars.*",
            ".*Mean of empty slice.*"]:
    warnings.filterwarnings("ignore", msg)


def train(logger):
    """
    perform the training routine for a given fold. saves plots and selected parameters to the experiment dir
    specified in the configs.
    """
    logger.info('performing training in {}D over fold {} on experiment {} with model {}'.format(
        cf.dim, cf.fold, cf.exp_dir, cf.model))

    net = model.net(cf, logger).cuda()
    if hasattr(cf, "optimizer") and cf.optimizer.lower() == "adam":
        logger.info("Using Adam optimizer.")
        optimizer = torch.optim.Adam(utils.parse_params_for_optim(net, weight_decay=cf.weight_decay,
                                                                   exclude_from_wd=cf.exclude_from_wd),
                                      lr=cf.learning_rate[0])
    else:
        logger.info("Using AdamW optimizer.")
        optimizer = torch.optim.AdamW(utils.parse_params_for_optim(net, weight_decay=cf.weight_decay,
                                                                   exclude_from_wd=cf.exclude_from_wd),
                                      lr=cf.learning_rate[0])


    if cf.dynamic_lr_scheduling:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=cf.scheduling_mode, factor=cf.lr_decay_factor,
                                                               patience=cf.scheduling_patience)

    model_selector = utils.ModelSelector(cf, logger)
    train_evaluator = Evaluator(cf, logger, mode='train')
    val_evaluator = Evaluator(cf, logger, mode=cf.val_mode)

    starting_epoch = 1

    # prepare monitoring
    monitor_metrics = utils.prepare_monitoring(cf)

    if cf.resume:
        checkpoint_path = os.path.join(cf.fold_dir, "last_checkpoint")
        starting_epoch, net, optimizer, monitor_metrics = \
            utils.load_checkpoint(checkpoint_path, net, optimizer)
        logger.info('resumed from checkpoint {} to epoch {}'.format(checkpoint_path, starting_epoch))


    logger.info('loading dataset and initializing batch generators...')
    batch_gen = data_loader.get_train_generators(cf, logger)

    # Prepare MLFlow
    best_loss = 1e3
    step = 1
    mlflow.log_artifacts(cf.exp_dir, "exp")

    for epoch in range(starting_epoch, cf.num_epochs + 1):

        logger.info('starting training epoch {}'.format(epoch))
        start_time = time.time()

        net.train()
        train_results_list = []
        for bix in range(cf.num_train_batches):
            batch = next(batch_gen['train'])
            tic_fw = time.time()
            results_dict = net.train_forward(batch)
            tic_bw = time.time()
            optimizer.zero_grad()
            results_dict['torch_loss'].backward()
            optimizer.step()
            print('\rtr. batch {0}/{1} (ep. {2}) fw {3:.2f}s / bw {4:.2f} s / total {5:.2f} s || '.format(
                bix + 1, cf.num_train_batches, epoch, tic_bw - tic_fw, time.time() - tic_bw,
                time.time() - tic_fw) + results_dict['logger_string'], flush=True, end="")
            train_results_list.append(({k:v for k,v in results_dict.items() if k != "seg_preds"}, batch["pid"]))
        print()

        _, monitor_metrics['train'] = train_evaluator.evaluate_predictions(train_results_list, monitor_metrics['train'])

        logger.info('generating training example plot.')
        utils.split_off_process(plot_batch_prediction, batch, results_dict, cf, outfile=os.path.join(
           cf.plot_dir, 'pred_example_{}_train.png'.format(cf.fold)))

        train_time = time.time() - start_time

        logger.info('starting validation in mode {}.'.format(cf.val_mode))
        with torch.no_grad():
            net.eval()
            if cf.do_validation:
                val_results_list = []
                val_predictor = Predictor(cf, net, logger, mode='val')
                for _ in range(batch_gen['n_val']):
                    batch = next(batch_gen[cf.val_mode])
                    if cf.val_mode == 'val_patient':
                        results_dict = val_predictor.predict_patient(batch)
                    elif cf.val_mode == 'val_sampling':
                        results_dict = net.train_forward(batch, is_validation=True)
                    val_results_list.append(({k:v for k,v in results_dict.items() if k != "seg_preds"}, batch["pid"]))

                _, monitor_metrics['val'] = val_evaluator.evaluate_predictions(val_results_list, monitor_metrics['val'])
                best_model_path = model_selector.run_model_selection(net, optimizer, monitor_metrics, epoch)
                # Save best model
                mlflow.log_artifacts(best_model_path, os.path.join("exp", os.path.basename(cf.fold_dir), 'best_checkpoint'))

            # Save logs and plots
            mlflow.log_artifacts(os.path.join(cf.exp_dir, "logs"), os.path.join("exp", 'logs'))
            mlflow.log_artifacts(cf.plot_dir, os.path.join("exp", os.path.basename(cf.plot_dir)))

            # update monitoring and prediction plots
            monitor_metrics.update({"lr":
                                        {str(g): group['lr'] for (g, group) in enumerate(optimizer.param_groups)}})

            # replace tboard metrics with MLFlow
            #logger.metrics2tboard(monitor_metrics, global_step=epoch)
            mlflow.log_metric('learning rate', optimizer.param_groups[0]['lr'], cf.num_epochs * cf.fold + epoch)
            for key in ['train', 'val']:
                for tag, val in monitor_metrics[key].items():
                    val = val[-1]  # maybe remove list wrapping, recording in evaluator?
                    if 'loss' in tag.lower() and not np.isnan(val):
                        mlflow.log_metric(f'{key}_{tag}', val, cf.num_epochs * cf.fold + epoch)
                    elif not np.isnan(val):
                        mlflow.log_metric(f'{key}_{tag}', val, cf.num_epochs * cf.fold + epoch)

            epoch_time = time.time() - start_time
            logger.info('trained epoch {}: took {} ({} train / {} val)'.format(
                epoch, utils.get_formatted_duration(epoch_time, "ms"), utils.get_formatted_duration(train_time, "ms"),
                utils.get_formatted_duration(epoch_time-train_time, "ms")))
            batch = next(batch_gen['val_sampling'])
            results_dict = net.train_forward(batch, is_validation=True)
            logger.info('generating validation-sampling example plot.')
            utils.split_off_process(plot_batch_prediction, batch, results_dict, cf, outfile=os.path.join(
                cf.plot_dir, 'pred_example_{}_val.png'.format(cf.fold)))

        # -------------- scheduling -----------------
        if cf.dynamic_lr_scheduling:
            scheduler.step(monitor_metrics["val"][cf.scheduling_criterion][-1])
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = cf.learning_rate[epoch-1]
    # Save whole experiment to MLFlow
    mlflow.log_artifacts(cf.exp_dir, "exp")

                
def test(logger):
    """
    perform testing for a given fold (or hold out set). save stats in evaluator.
    """
    logger.info('starting testing model of fold {} in exp {}'.format(cf.fold, cf.exp_dir))
    net = model.net(cf, logger).cuda()
    test_predictor = Predictor(cf, net, logger, mode='test')
    test_evaluator = Evaluator(cf, logger, mode='test')
    batch_gen = data_loader.get_test_generator(cf, logger)
    test_results_list = test_predictor.predict_test_set(batch_gen, return_results=True)
    test_evaluator.evaluate_predictions(test_results_list)
    test_evaluator.score_test_df()


def predict(logger, save_seg=False):
    """
    perform prediction for a given fold (or hold out set). save stats in evaluator.
    """
    net = model.net(cf, logger).cuda()
    test_predictor = Predictor(cf, net, logger, mode='pred')
    logger.info(f'starting prediction on model of fold {cf.fold} (epoch {np.max(test_predictor.epoch_ranking)}) in exp {cf.exp_dir}')
    output_path = os.path.join(cf.output_dir, 'pp_img.npy')
    _, itk_origin, itk_spacing, itk_shape = preprocessing.preprocess_image(cf.patient_path, output_path, save_lungs_mask=True)
    cf.patient_path = output_path
    batch_gen = data_loader.get_pred_generator(cf, logger)
    weight_path = os.path.join(cf.fold_dir, '{}_best_checkpoint'.format(np.max(test_predictor.epoch_ranking)), 'params.pth')
    net.load_state_dict(torch.load(weight_path))
    test_results_list = test_predictor.predict_patient(next(batch_gen['pred']))
    if save_seg:
        # Build segmentation mask
        seg = test_results_list['seg_preds'][0][0]
        seg_mask = np.zeros(seg.shape).astype(np.uint8)
        seg_tresh = cf.min_det_thresh  # minimum confidence value to select predictions
        seg_mask[seg > seg_tresh] = 1
        # Swap seg axes to match original image
        seg_mask = np.swapaxes(seg_mask, 0, 2)
        seg_mask = np.swapaxes(seg_mask, 1, 2)
        # Resample segmentation to original shape
        seg_mask, seg_spacing = preprocessing.resample_array_to_shape(seg_mask, cf.target_spacing, target_shape=itk_shape)
        seg_mask[seg_mask <= 0.5] = 0
        seg_mask[seg_mask > 0] = 1
        seg_mask = seg_mask.astype(np.uint8)
        # Save segmentation to NRRD file
        data_utils.write_itk(seg_mask, itk_origin, itk_spacing, os.path.join(cf.output_dir, 'nodules_seg.nrrd'))


if __name__ == '__main__':
    stime = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str,  default='train_test',
                        help='one out of: train / test / train_test / analysis / create_exp')
    parser.add_argument('-f','--folds', nargs='+', type=int, default=None,
                        help='None runs over all folds in CV. otherwise specify list of folds.')
    parser.add_argument('--exp_dir', type=str, default='/path/to/experiment/directory',
                        help='path to experiment dir. will be created if non existent.')
    parser.add_argument('--server_env', default=False, action='store_true',
                        help='change IO settings to deploy models on a cluster.')
    parser.add_argument('--data_dest', type=str, default=None, help="path to final data folder if different from config.")
    parser.add_argument('--use_stored_settings', default=False, action='store_true',
                        help='load configs from existing exp_dir instead of source dir. always done for testing, '
                             'but can be set to true to do the same for training. useful in job scheduler environment, '
                             'where source code might change before the job actually runs.')
    parser.add_argument('--resume', action="store_true", default=False,
                        help='if given, resume from checkpoint(s) of the specified folds.')
    parser.add_argument('--exp_source', type=str, default='experiments/toy_exp',
                        help='specifies, from which source experiment to load configs and data_loader.')
    parser.add_argument('--no_benchmark', action='store_true', help="Do not use cudnn.benchmark.")
    parser.add_argument('--cuda_device', type=int, default=0, help="Index of CUDA device to use.")
    parser.add_argument('-d', '--dev', default=False, action='store_true', help="development mode: shorten everything")
    parser.add_argument('--mlflow_uri', type=str, required=False, help="MLFlow server URI")
    parser.add_argument('--mlflow_artifacts_uri', type=str, required=False, help="MLFlow artifacts URI")
    parser.add_argument('--mlflow_experiment_id', type=str, required=False, default="MDT experiment", help="MLFlow experiment ID")
    parser.add_argument('--mlflow_run_id', type=str, required=False, help="MLFlow run ID")
    parser.add_argument('--mlflow_model_name', type=str, required=False, help="Name of MLFlow Model Registry model to retrieve.")
    parser.add_argument('--patient_path', type=str, required=False,
                         help='specifies the npy file of the current patient.')
    parser.add_argument('--output_dir', type=str, required=False, default="pred_output", 
                         help='Output directory')
    parser.add_argument('--large_model_support', action="store_true", default=False,
                        help='If given, enable Large Model Support (LMS) during training.')

    args = parser.parse_args()
    folds = args.folds

    torch.backends.cudnn.benchmark = not args.no_benchmark

    # MLFlow setup
    if args.mlflow_uri is not None:
        mlflow.set_tracking_uri(args.mlflow_uri)

    # Enable Large Model Support (LMS)
    if args.large_model_support is True:
        torch.cuda.set_enabled_lms(True)

    if args.mode == 'train' or args.mode == 'train_test':

        cf = utils.prep_exp(args.exp_source, args.exp_dir, args.server_env, args.use_stored_settings)
        if args.dev:
            folds = [0,1]
            cf.batch_size, cf.num_epochs, cf.min_save_thresh, cf.save_n_models = 3 if cf.dim==2 else 1, 1, 0, 2
            cf.num_train_batches, cf.num_val_batches, cf.max_val_patients = 5, 1, 1
            cf.test_n_epochs =  cf.save_n_models
            cf.max_test_patients = 2

        cf.data_dest = args.data_dest
        logger = utils.get_logger(cf.exp_dir, cf.server_env)
        logger.info("cudnn benchmark: {}, deterministic: {}.".format(torch.backends.cudnn.benchmark,
                                                                     torch.backends.cudnn.deterministic))
        logger.info("sending tensors to CUDA device: {}.".format(torch.cuda.get_device_name(args.cuda_device)))
        data_loader = utils.import_module('dl', os.path.join(args.exp_source, 'data_loader.py'))
        model = utils.import_module('mdt_model', cf.model_path)
        logger.info("loaded model from {}".format(cf.model_path))
        if folds is None:
            folds = range(cf.n_cv_splits)

        # MLFLow new experiment
        try:
            if args.mlflow_artifacts_uri is not None:
                exp_id = mlflow.create_experiment(args.mlflow_experiment_id, artifact_location=args.mlflow_artifacts_uri)
            else:
                exp_id = mlflow.create_experiment(args.mlflow_experiment_id)
        except:
            exp_id = mlflow.set_experiment(args.mlflow_experiment_id)

        with torch.cuda.device(args.cuda_device):
            with mlflow.start_run(experiment_id=exp_id) as run:
                # Log tags to MLFlow
                mlflow.set_tag("Machine", platform.system())
                mlflow.set_tag("Release", platform.release())
                try:
                    mlflow.set_tag("GPU", torch.cuda.get_device_name(args.cuda_device))
                except Exception:
                    mlflow.set_tag("GPU", "None")
                # Log hyper-params to MLFlow
                mlflow.log_param("Dimension", f"{cf.dim}D")
                mlflow.log_param("Model", cf.model)
                mlflow.log_param("Resume", args.resume)
                mlflow.log_param("Epochs", cf.num_epochs)
                mlflow.log_param("Optimizer", cf.optimizer)
                mlflow.log_param("Output Classes", cf.head_classes - 1)  # -1 for bg
                mlflow.log_param("Batch size", cf.batch_size)
                mlflow.log_param("Nb RPN feature maps", cf.n_rpn_features)
                mlflow.log_param("Model Selection Criteria", ', '.join(cf.model_selection_criteria))
                for fold in folds:
                    cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(fold))
                    cf.fold = fold
                    cf.resume = args.resume
                    if not os.path.exists(cf.fold_dir):
                        os.mkdir(cf.fold_dir)
                    logger.set_logfile(fold=fold)
                    # Start training
                    train(logger)
            cf.resume = False
            if args.mode == 'train_test':
                test(logger)

    elif args.mode == 'test':

        # Tries to reach experiment locally
        if not os.path.exists(args.exp_dir):
            os.makedirs(args.exp_dir)
            # Load exp from MLFlow
            mlflow_tracking = mlflow.tracking.MlflowClient(tracking_uri=args.mlflow_uri)
            if args.mlflow_run_id is not None:
                args.exp_dir = mlflow_tracking.download_artifacts(args.mlflow_run_id, "exp", os.path.abspath(args.exp_dir))
            else:
                # Get last production model (experiment) from MLFlow model registry
                run_id = None
                version = 0
                for mv in mlflow_tracking.search_model_versions(f"name='{args.mlflow_model_name}'"):
                    if dict(mv)['current_stage'] == 'Production' and int(dict(mv)['version']) > version:
                        run_id = dict(mv)['run_id']
                        version = dict(mv)['version']
                try:
                    print(f"Loading production model from MLFLow (version {version})")
                    args.exp_dir = mlflow_tracking.download_artifacts(run_id, "exp", os.path.abspath(args.exp_dir))
                except Exception:
                    raise(Exception("ERROR: No model in production!"))
        cf = utils.prep_exp(args.exp_source, args.exp_dir, args.server_env, is_training=False, use_stored_settings=True)

        if args.dev:
            folds = [0,1]
            cf.test_n_epochs = 2; cf.max_test_patients = 2

        cf.data_dest = args.data_dest
        logger = utils.get_logger(cf.exp_dir, cf.server_env)
        data_loader = utils.import_module('dl', os.path.join(args.exp_source, 'data_loader.py'))
        model = utils.import_module('mdt_model', cf.model_path)
        logger.info("loaded model from {}".format(cf.model_path))
        if folds is None:
            folds = range(cf.n_cv_splits)

        with torch.cuda.device(args.cuda_device):
            for fold in folds:
                cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(fold))
                cf.fold = fold
                logger.set_logfile(fold=fold)
                test(logger)

    elif args.mode == 'predict':

        # Tries to reach experiment locally
        if not os.path.exists(args.exp_dir):
            os.makedirs(args.exp_dir)
            # Load exp from MLFlow
            mlflow_tracking = mlflow.tracking.MlflowClient(tracking_uri=args.mlflow_uri)
            if args.mlflow_run_id is not None:
                args.exp_dir = mlflow_tracking.download_artifacts(args.mlflow_run_id, "exp", os.path.abspath(args.exp_dir))
            else:
                # Get last production model (experiment) from MLFlow model registry
                run_id = None
                version = 0
                for mv in mlflow_tracking.search_model_versions(f"name='{args.mlflow_model_name}'"):
                    if dict(mv)['current_stage'] == 'Production' and int(dict(mv)['version']) > version:
                        run_id = dict(mv)['run_id']
                        version = int(dict(mv)['version'])
                try:
                    print(f"Loading production model from MLFLow (version {version})")
                    args.exp_dir = mlflow_tracking.download_artifacts(run_id, "exp", os.path.abspath(args.exp_dir))
                except Exception:
                    raise(Exception("ERROR importing model from MLFlow"))
        cf = utils.prep_exp(args.exp_source, args.exp_dir, args.server_env, is_training=False, use_stored_settings=True)

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        cf.output_dir = args.output_dir

        cf.data_dest = args.data_dest
        logger = utils.get_logger(cf.exp_dir, cf.server_env)
        data_loader = utils.import_module('dl', os.path.join(args.exp_source, 'data_loader.py'))
        preprocessing = utils.import_module('pp', os.path.join(args.exp_source, 'preprocessing.py'))
        data_utils = utils.import_module('du', os.path.join(args.exp_source, 'data_utils.py'))
        model = utils.import_module('mdt_model', cf.model_path)
        logger.info("loaded model from {}".format(cf.model_path))
        if folds is None:
            folds = range(cf.n_cv_splits)

        with torch.cuda.device(args.cuda_device):
            for fold in folds:
                cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(fold))
                cf.fold = fold
                logger.set_logfile(fold=fold)
                cf.patient_path=args.patient_path
                predict(logger, save_seg=True)

    # load raw predictions saved by predictor during testing, run aggregation algorithms and evaluation.
    elif args.mode == 'analysis':
        cf = utils.prep_exp(args.exp_source, args.exp_dir, args.server_env, is_training=False, use_stored_settings=True)
        logger = utils.get_logger(cf.exp_dir, cf.server_env)

        if args.dev:
            cf.test_n_epochs = 2

        if cf.hold_out_test_set and cf.ensemble_folds:
            # create and save (unevaluated) predictions across all folds
            predictor = Predictor(cf, net=None, logger=logger, mode='analysis')
            results_list = predictor.load_saved_predictions(apply_wbc=True)
            utils.create_csv_output([(res_dict["boxes"], pid) for res_dict, pid in results_list], cf, logger)
            logger.info('starting evaluation...')
            cf.fold = 'overall_hold_out'
            evaluator = Evaluator(cf, logger, mode='test')
            evaluator.evaluate_predictions(results_list)
            evaluator.score_test_df()

        else:
            fold_dirs = sorted([os.path.join(cf.exp_dir, f) for f in os.listdir(cf.exp_dir) if
                         os.path.isdir(os.path.join(cf.exp_dir, f)) and f.startswith("fold")])
            if folds is None:
                folds = range(cf.n_cv_splits)
            for fold in folds:
                cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(fold))
                cf.fold = fold
                logger.set_logfile(fold=fold)
                if cf.fold_dir in fold_dirs:
                    predictor = Predictor(cf, net=None, logger=logger, mode='analysis')
                    results_list = predictor.load_saved_predictions(apply_wbc=True)
                    logger.info('starting evaluation...')
                    evaluator = Evaluator(cf, logger, mode='test')
                    evaluator.evaluate_predictions(results_list)
                    evaluator.score_test_df()
                else:
                    logger.info("Skipping fold {} since no model parameters found.".format(fold))

    # create experiment folder and copy scripts without starting job.
    # useful for cloud deployment where configs might change before job actually runs.
    elif args.mode == 'create_exp':
        cf = utils.prep_exp(args.exp_source, args.exp_dir, args.server_env, use_stored_settings=False)
        logger = utils.get_logger(cf.exp_dir)
        logger.info('created experiment directory at {}'.format(cf.exp_dir))

    else:
        raise RuntimeError('mode specified in args is not implemented...')


    t = utils.get_formatted_duration(time.time() - stime)
    logger.info("{} total runtime: {}".format(os.path.split(__file__)[1], t))
    del logger
