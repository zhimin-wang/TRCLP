# Copyright (c) Wang Zhimin zm.wang@buaa.edu.cn 2024/10/28 All Rights Reserved.

import sys
sys.path.append('../')
import os
import shutil
import time

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib.patches as mpatches


from utils import (Load_20_P_allY_TrainingData, Load_20_P_allY_TrainingDataForDemo,
                   Load_20_P_allY_TestData, RemakeDir, MakeDir, SeedTorch,
                   plot_confusion_matrix, SupConLoss, AverageMeter)
from models import weight_init
from models.TRCLPModels import *
import datetime
matplotlib.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda")
print(device)
print("torch.cuda.device_count(): ", torch.cuda.device_count())

use_multi_gpu = False


def pretraining(args, epoch, model, train_loader, optimizer, exp_lr_scheduler, criterion, losses, step_num, num_epochs):
    """Pretraining function for the models."""
    # Set models to training mode and move to device
    for num in range(3):
        model[num].cuda()
        model[num].train()

    for i, (features, aug_features, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        features = features.reshape(-1, args.inputSize).to(device)
        aug_features = aug_features.reshape(-1, args.inputSize).to(device)
        feature_both = torch.cat([features, aug_features], dim=0)
        batch_size = labels.shape[0]

        # Use the last output as label
        feature_len = int(args.eyeFeatureSize / 2)
        labels = labels[:, feature_len - 1:feature_len].view(-1).to(device)
        loss_list = []

        for model_id in range(3):
            # Forward pass
            outputs = model[model_id](feature_both)

            # Compute loss
            f1, f2 = torch.split(outputs, [batch_size, batch_size], dim=0)
            con_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            loss = criterion(con_features, labels)
            loss = loss.mean() if use_multi_gpu else loss

            # Update metric
            losses[model_id].update(loss.item(), batch_size)

            # Backward and optimize
            optimizer[model_id].zero_grad()
            loss.backward()
            optimizer[model_id].step()
            loss_list.append(loss)

        # Output the loss at specified intervals
        if (i + 1) % int(step_num / args.lossFrequency) == 0:
            for model_id in range(3):
                print('Epoch [{}/{}], Step [{}/{}], model_id={} Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, step_num, model_id, loss_list[model_id].item()))


def training(args, epoch, model, classifier, train_loader, optimizer, exp_lr_scheduler, criterion, losses, step_num, num_epochs):
    """Training function for the classifier."""
    # Set models to evaluation mode and move to device
    for num in range(3):
        model[num].cuda()
        model[num].eval()

    classifier.cuda()
    classifier.train()

    for i, (features, _, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        features = features.reshape(-1, args.inputSize).to(device)
        batch_size = labels.shape[0]

        # Use the last output as label
        feature_len = int(args.eyeFeatureSize / 2)
        labels = labels[:, feature_len - 1:feature_len].view(-1).to(device)

        # Forward pass
        output_1 = []
        with torch.no_grad():
            for num in range(3):
                output = model[num](features)
                output_1.append(output.to(device) if use_multi_gpu else output)

        # Concatenate outputs
        out_1 = torch.cat(output_1, dim=1)
        output_2 = classifier(out_1.to(device) if use_multi_gpu else out_1)
        loss = criterion(output_2, labels)
        loss = loss.mean() if use_multi_gpu else loss

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Output the loss at specified intervals
        if (i + 1) % int(step_num / args.lossFrequency) == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, step_num, loss.item()))


def filter_predictions(current_test_label, predictions_no_filter):
    """Filter function to smooth predictions."""
    pre_data_list = []
    pre_data_time = 2 * 25  # 25Hz
    drop_pre_time = 2 * 25
    post_data_list = []
    post_data_time = 1 * 25  # 25Hz
    drop_post_time = 1 * 25
    pre_threshold_ratio = 0.8
    post_threshold_ratio = 0.8

    for j in range(predictions_no_filter.shape[0]):
        task = predictions_no_filter[j]

        max_pre_task_id = 0
        max_pre_task_ratio = 0
        max_pre_task_num = 0

        max_post_task_id = 0
        max_post_task_ratio = 0
        max_post_task_num = 0

        next_id = j + post_data_time
        if next_id < predictions_no_filter.shape[0]:
            post_data_list.append(predictions_no_filter[next_id])

        # Maintain pre_data_list, keep data of previous 4 seconds, count the proportion.
        if len(pre_data_list) > pre_data_time:
            del pre_data_list[0]

        for task_id in range(4):
            if pre_data_list.count(task_id) > max_pre_task_num:
                max_pre_task_id = task_id
                max_pre_task_num = pre_data_list.count(task_id)
                max_pre_task_ratio = max_pre_task_num / len(pre_data_list)

            if post_data_list.count(task_id) > max_post_task_num:
                max_post_task_id = task_id
                max_post_task_num = post_data_list.count(task_id)
                max_post_task_ratio = max_post_task_num / len(post_data_list)

        reliable_pre_list = False
        reliable_post_list = True

        if len(pre_data_list) >= drop_pre_time and max_pre_task_ratio >= pre_threshold_ratio:
            reliable_pre_list = True

        if len(pre_data_list) >= drop_pre_time and len(post_data_list) >= drop_post_time:
            if max_post_task_ratio > post_threshold_ratio and max_post_task_id != task:
                if (max_pre_task_ratio >= pre_threshold_ratio and max_pre_task_id != task) or max_pre_task_ratio < pre_threshold_ratio:
                    predictions_no_filter[j] = max_post_task_id  # Modify current value to max_post_task_id
            elif (max_post_task_ratio < post_threshold_ratio) and (max_pre_task_ratio >= pre_threshold_ratio and max_pre_task_id != task):
                predictions_no_filter[j] = max_pre_task_id  # Modify current value to max_pre_task_id

        pre_data_list.append(predictions_no_filter[j])

        # Maintain post_data_list, keep data of next 1 second, count the proportion.
        if len(post_data_list) > post_data_time:
            del post_data_list[0]

    return predictions_no_filter


cur_user = 0
cur_num = 0
cur_total = 0


def data_smooth(test_label, test_size, test_y, pred_y, pred_dir, args):
    """Function to smooth data and visualize results."""
    global cur_user
    global cur_num
    global cur_total
    start_id = -1
    end_id = -1
    pre_user = -1
    pre_scene = -1
    for item_id, test_item in enumerate(test_label):
        if test_item[0] != pre_user or test_item[1] != pre_scene or item_id == test_size - 1:
            if item_id != test_size - 1:
                end_id = item_id - 1
            else:
                end_id = item_id

            if args.onlyFilter == 1:
                if start_id != -1:
                    pred_y[start_id:end_id + 1] = filter_predictions(
                        test_label[start_id:end_id + 1], pred_y[start_id:end_id + 1])
            else:
                if start_id != -1:
                    # Calculate each person
                    user_id = test_label[end_id][0]
                    scene_id = test_label[end_id][1]

                    prediction_results = np.zeros(shape=(end_id - start_id + 1, 4))
                    prediction_results[:, 0] = test_y[start_id:end_id + 1].reshape(-1,)
                    prediction_results[:, 1] = pred_y[start_id:end_id + 1].reshape(-1,)
                    plt.close()
                    plt.cla()
                    plt.clf()
                    figure, ax = plt.subplots(figsize=(17, 8))

                    plt.xlabel("Time (s)", fontsize=12, fontweight='bold')
                    my_x_ticks = np.arange(0, 160, 10)
                    plt.xticks(my_x_ticks, fontsize=12, fontweight='normal', fontproperties='arial')
                    plt.minorticks_on()

                    my_y_ticks = np.arange(1, 4, 1)
                    plt.yticks(my_y_ticks, labels=['After optimizing', 'Recognized types', 'Ground truth'],
                               fontsize=12, fontweight='bold')
                    plt.ylim(0, 6)

                    colormap1 = (143 / 255.0, 170 / 255.0, 220 / 255.0)
                    colormap2 = (251 / 255.0, 128 / 255.0, 104 / 255.0)
                    colormap3 = (128 / 255.0, 230 / 255.0, 155 / 255.0)
                    colormap4 = (255 / 255.0, 200 / 255.0, 44 / 255.0)

                    color_set = (colormap1, colormap2, colormap3, colormap4)

                    x_list = [[] for _ in range(4)]
                    y_list = [[] for _ in range(4)]

                    current_test_label = test_label[start_id:end_id + 1]
                    segment_start = current_test_label[0, 2]
                    segment_end = -1

                    capture_text = []
                    capture_x = []
                    capture_y = []
                    capture_height = 3.2
                    delta = 0.2
                    base_height = 3

                    # Ground Truth
                    for j in range(test_y[start_id:end_id + 1].shape[0]):

                        # New task
                        if (j != 0 and current_test_label[j, 3] - current_test_label[j - 1, 3] > 1000) \
                                or j == test_y[start_id:end_id + 1].shape[0] - 1 \
                                or (j != 0 and current_test_label[j, 4] != current_test_label[j - 1, 4]):
                            if j == test_y[start_id:end_id + 1].shape[0] - 1:
                                segment_end = current_test_label[j, 2]
                            else:
                                segment_end = current_test_label[j - 1, 2]
                            # Draw capture
                            capture_x.append(segment_start)
                            capture_y.append(capture_height)
                            cap_text = current_test_label[j - 1, 4]
                            if len(current_test_label[j - 1, 4]) > 100:
                                cap_text = current_test_label[j - 1, 4][0:50] + "..."
                            capture_text.append(cap_text)
                            if capture_height + delta > base_height + 3.09:
                                delta = -0.2
                            if capture_height + delta < base_height + 0.19:
                                delta = 0.2
                            capture_height += delta

                            segment_start = current_test_label[j, 2]

                        task = test_y[start_id:end_id + 1][j]  # Current task
                        x_list[task].append(current_test_label[j, 2])  # Video time
                        y_list[task].append(3)

                    for text, c_x, c_y in zip(capture_text, capture_x, capture_y):
                        plt.annotate(text, (c_x, c_y), fontsize=10, fontweight='bold')

                    for i in range(4):
                        color = color_set[i] if i < 2 else color_set[3] if i == 2 else color_set[2]
                        plt.scatter(x_list[i], y_list[i], color=color, marker="|", s=500)

                    x_list = [[] for _ in range(4)]
                    y_list = [[] for _ in range(4)]

                    for j in range(pred_y[start_id:end_id + 1].shape[0]):
                        task = pred_y[start_id:end_id + 1][j]
                        x_list[task].append(current_test_label[j, 2])  # Video time
                        prediction_results[j, 3] = current_test_label[j, 2]
                        y_list[task].append(2)

                    correct1 = (test_y[start_id:end_id + 1] == pred_y[start_id:end_id + 1]).sum()
                    accuracy1 = correct1 / (end_id - start_id + 1) * 100

                    pred_y[start_id:end_id + 1] = filter_predictions(
                        current_test_label, pred_y[start_id:end_id + 1])  # Filter the noise
                    prediction_results[:, 2] = pred_y[start_id:end_id + 1].reshape(-1,)
                    for j in range(pred_y[start_id:end_id + 1].shape[0]):
                        task = pred_y[start_id:end_id + 1][j]
                        x_list[task].append(current_test_label[j, 2])  # Video time
                        y_list[task].append(1)

                    for i in range(4):
                        color = color_set[i] if i < 2 else color_set[3] if i == 2 else color_set[2]
                        plt.scatter(x_list[i], y_list[i], color=color, marker="|", s=500)

                    path1 = mpatches.Patch(color=color_set[0], label="Fixating")
                    path2 = mpatches.Patch(color=color_set[1], label="Observing")
                    path3 = mpatches.Patch(color=color_set[2], label="Tracking")
                    path4 = mpatches.Patch(color=color_set[3], label="Free exploration")
                    handles = [path1, path2, path3, path4]
                    ax.tick_params(direction='in')
                    ax.legend(handles=handles, mode="expand", ncol=4, borderaxespad=0,
                              prop=dict(weight='bold', size=14), markerscale=10)

                    correct2 = (test_y[start_id:end_id + 1] == pred_y[start_id:end_id + 1]).sum()
                    accuracy2 = correct2 / (end_id - start_id + 1) * 100

                    ax.set_title("user_{:d}_Scene_{:d}.  Accuracy: {:.1f}->{:.1f}%"
                                 .format(user_id, scene_id, accuracy1, accuracy2),
                                 fontweight='bold', fontsize=16)

                    if cur_user == user_id:
                        cur_num += correct2
                        cur_total += (end_id - start_id + 1)
                    else:
                        cur_user = user_id
                        cur_num = correct2
                        cur_total = (end_id - start_id + 1)
                    sub_file = pred_dir + "{:02d}_{:02d}/".format(user_id, scene_id)
                    sub_name = "{:02d}_{:02d}_".format(user_id, scene_id)
                    MakeDir(sub_file)

                    plt.savefig(sub_file + sub_name + 'result_visualization.jpg')

                    np.savetxt(sub_file + sub_name + 'predictions.txt', prediction_results, fmt='%.02f')
                    plot_confusion_matrix(test_y[start_id:end_id + 1], pred_y[start_id:end_id + 1],
                                          ['FS', 'OS', 'FE', 'TM'], "Confusion matrix",
                                          sub_file + sub_name + 'Confusion matrix.jpg', True)

            pre_user = test_item[0]
            pre_scene = test_item[1]
            start_id = item_id


def main(args):
    """Main function to train and test the model."""
    print('\n==> Creating the model...')

    # Define three models
    model = []
    for i in range(3):
        model_instance = sinCNN_BiGRU(args.eyeFeatureSize, i)
        model_instance.apply(weight_init)
        if use_multi_gpu:
            model_instance = nn.DataParallel(model_instance, device_ids=[0, 1])
        else:
            model_instance = nn.DataParallel(model_instance)
        model.append(model_instance)

    classifier = comb_FC(args.eyeFeatureSize, args.headFeatureSize, args.gwFeatureSize, args.numClasses)
    classifier.apply(weight_init)
    if use_multi_gpu:
        classifier = nn.DataParallel(classifier, device_ids=[0, 1])
    else:
        classifier = nn.DataParallel(classifier)

    # Define loss functions
    criterion_pretrain = SupConLoss(temperature=args.temp)
    criterion_classifier = nn.CrossEntropyLoss()

    if args.pretrainFlag == 1 or args.trainClassifierFlag == 1:
        # Load the training data
        strip = 5  # Keep the task change flag
        if args.demo == 1:
            train_loader = Load_20_P_allY_TrainingDataForDemo(strip, args.datasetDir, args.batchSize)
        else:
            train_loader = Load_20_P_allY_TrainingData(strip, args.datasetDir, args.batchSize, args.augmode, args.trainClassifierFlag)

    start_time = datetime.datetime.now()
    # Train the model
    if args.pretrainFlag == 1:
        # Optimizer and learning rate scheduler
        lr = args.learningRate

        optimizer = []
        exp_lr_scheduler = []
        for i in range(3):
            opt = torch.optim.Adam(model[i].parameters(), lr=lr, weight_decay=args.weightDecay)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.gamma, last_epoch=-1)
            optimizer.append(opt)
            exp_lr_scheduler.append(scheduler)

        # Training start epoch
        start_epoch = 0
        # Remake checkpoint directory
        RemakeDir(args.checkpoint)

        # Training
        local_time = time.asctime(time.localtime(time.time()))
        print('\nPretraining starts at ' + local_time)

        # The number of training steps in an epoch
        step_num = len(train_loader)
        num_epochs = args.encoderEpochs

        for epoch in range(start_epoch, num_epochs):
            print('\nEpoch: {} | LR: {:.16f}'.format(epoch + 1, lr))
            losses = [AverageMeter() for _ in range(3)]

            pretraining(args, epoch, model, train_loader, optimizer, exp_lr_scheduler, criterion_pretrain, losses, step_num, num_epochs)
            # Adjust learning rate
            for scheduler in exp_lr_scheduler:
                scheduler.step()

            end_time = datetime.datetime.now()
            total_training_time = (end_time - start_time).seconds / 60
            print('\nEpoch [{}/{}], Total Training Time: {:.2f} min'.format(epoch + 1, num_epochs, total_training_time))

            # Save the checkpoint
            if (epoch + 1) % args.interval == 0:
                for num in range(3):
                    save_path = os.path.join(args.checkpoint, "checkpoint_Three_epoch_{}_{}.tar".format(str(epoch + 1).zfill(3), num))
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model[num].state_dict() if not use_multi_gpu else model[num].module.state_dict(),
                        'optimizer_state_dict': optimizer[num].state_dict(),
                        'loss': losses[num].avg,
                        'lr': lr,
                    }, save_path)

        for num in range(3):
            save_path = os.path.join(args.checkpoint, "last_path_{}.tar".format(num))
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model[num].state_dict() if not use_multi_gpu else model[num].module.state_dict(),
                'optimizer_state_dict': optimizer[num].state_dict(),
                'loss': losses[num].avg,
                'lr': lr,
            }, save_path)

        local_time = time.asctime(time.localtime(time.time()))
        print('\nTraining ends at ' + local_time)
    epochs = args.classiferEpochs

    encoder_epochs = [15, 30]
    find_flag = False
    for epo in encoder_epochs:
        # Train Classifier
        if args.trainClassifierFlag == 1:
            for num in range(3):
                # Load the pretrained model
                checkpoint_path = os.path.join(args.checkpoint, "checkpoint_Three_epoch_{}_{}.tar".format(str(epo).zfill(3), num))
                if device == torch.device('cuda'):
                    checkpoint = torch.load(checkpoint_path)
                    print('\nDevice: GPU')
                else:
                    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
                    print('\nDevice: CPU')

                model[num].load_state_dict(checkpoint['model_state_dict'])

            # Optimizer and learning rate scheduler
            lr = args.learningRate
            optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=args.weightDecay)
            exp_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma, last_epoch=-1)

            # Training start epoch
            start_epoch = 0

            local_time = time.asctime(time.localtime(time.time()))
            print('\nTraining Classifier starts at ' + local_time)

            # The number of training steps in an epoch
            step_num = len(train_loader)
            num_epochs = epochs

            for epoch in range(start_epoch, num_epochs):
                print('\nEpoch: {} | LR: {:.16f}'.format(epoch + 1, lr))

                losses = AverageMeter()
                training(args, epoch, model, classifier, train_loader, optimizer, exp_lr_scheduler, criterion_classifier, losses, step_num, num_epochs)

                # Adjust learning rate
                exp_lr_scheduler.step()
                end_time = datetime.datetime.now()
                total_training_time = (end_time - start_time).seconds / 60
                print('\nEpoch [{}/{}], Total Training Time: {:.2f} min'.format(epoch + 1, num_epochs, total_training_time))

                # Save the checkpoint
                if (epoch + 1) % args.interval == 0:
                    save_path = os.path.join(args.checkpoint, "checkpoint_FC_epoch_{}_{}.tar".format(str(epo).zfill(3), str(epoch + 1).zfill(3)))
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': classifier.state_dict() if not use_multi_gpu else classifier.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': losses.avg,
                        'lr': lr,
                    }, save_path)

        if args.test == 1:
            # Test all the existing models
            if os.path.isdir(args.checkpoint):
                filelist = os.listdir(args.checkpoint)
                checkpoints = []
                checkpoint_num = 0
                for name in filelist:
                    # Checkpoints are stored as tar files
                    if args.savePrd:
                        expected_name = 'checkpoint_FC_epoch_{}_{}.tar'.format(str(args.runEncoder).zfill(3), str(args.runClassifer).zfill(3))
                        if name != expected_name:
                            continue
                        find_flag = True
                    if name.startswith("checkpoint_FC_epoch_{}_" .format(str(epo).zfill(3))):
                        print(name)
                        checkpoints.append(name)
                        checkpoint_num += 1

                epoch_list = []
                accuracy_list = []

                # Test the checkpoints
                if checkpoint_num:
                    print('\nCheckpoint Number : {}'.format(checkpoint_num))
                    checkpoints.sort()

                    strip = 6  # Delete the task change flag
                    # Load the test data
                    test_loader, test_label = Load_20_P_allY_TestData(strip, args.datasetDir, args.batchSize)
                    # Load the test labels
                    feature_len = int(args.eyeFeatureSize / 2)
                    test_y = test_label[:, strip + feature_len - 1:strip + feature_len].astype(np.int32).reshape(-1)
                    test_size = test_y.shape[0]
                    # Save the predictions
                    if args.savePrd:
                        pred_dir = args.prdDir
                        RemakeDir(pred_dir)
                    local_time = time.asctime(time.localtime(time.time()))
                    print('\nTest starts at ' + local_time)

                    for num in range(3):
                        # Load the pretrained model
                        checkpoint_path = os.path.join(args.checkpoint, "checkpoint_Three_epoch_{}_{}.tar".format(str(epo).zfill(3), num))
                        if device == torch.device('cuda'):
                            checkpoint = torch.load(checkpoint_path)
                            print('\nDevice: GPU')
                        else:
                            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
                            print('\nDevice: CPU')

                        model[num].load_state_dict(checkpoint['model_state_dict'])

                    for name in checkpoints:
                        print("\n==> Test checkpoint : {}".format(name))
                        checkpoint_path = os.path.join(args.checkpoint, name)
                        if device == torch.device('cuda'):
                            checkpoint = torch.load(checkpoint_path)
                            print('\nDevice: GPU')
                        else:
                            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
                            print('\nDevice: CPU')

                        if use_multi_gpu:
                            state_dict = checkpoint['model_state_dict']
                            from collections import OrderedDict
                            new_state_dict = OrderedDict()

                            for k, v in state_dict.items():
                                if 'module' not in k:
                                    k = 'module.' + k
                                new_state_dict[k] = v

                            classifier.load_state_dict(new_state_dict)
                        else:
                            classifier.load_state_dict(checkpoint['model_state_dict'])
                        epoch = checkpoint['epoch']
                        # The model's predictions
                        pred_y = []

                        # Set models to evaluation mode
                        for num in range(3):
                            model[num].cuda()
                            model[num].eval()

                        classifier.cuda()
                        classifier.eval()

                        start_time = datetime.datetime.now()
                        for i, (features, _) in enumerate(test_loader):
                            # Move tensors to the configured device
                            features = features.reshape(-1, args.inputSize).to(device)

                            # Forward pass
                            output_1 = []
                            with torch.no_grad():
                                for num in range(3):
                                    output_1.append(model[num](features))

                            # Concatenate outputs
                            out_1 = torch.cat(output_1, dim=1)
                            output_2 = classifier(out_1)

                            _, predictions = torch.max(output_2.data, 1)

                            # Save the predictions
                            predictions_npy = predictions.cpu().detach().numpy()

                            if len(pred_y) > 0:
                                pred_y = np.concatenate((pred_y, predictions_npy.reshape(-1)))
                            else:
                                pred_y = predictions_npy.reshape(-1)

                        end_time = datetime.datetime.now()
                        # Average predicting time for a single sample.
                        avg_time = (end_time - start_time).seconds * 1000 / test_size
                        print('\nAverage prediction time: {:.8f} ms'.format(avg_time))

                        # Calculate the prediction accuracy
                        chance_accuracy = 1 / args.numClasses * 100
                        print('Chance Level Accuracy: {:.1f}%'.format(chance_accuracy))
                        pred_y = pred_y.reshape(-1)
                        correct = (test_y == pred_y).sum()
                        accuracy = correct / test_size * 100
                        print('Epoch: {}, Single Window Prediction Accuracy: {:.1f}%'.format(epoch, accuracy))
                        epoch_list.append(epoch)
                        accuracy_list.append(accuracy)

                        MakeDir(args.prdDir + 'predictions_epoch_{}/'.format(str(epoch).zfill(3)))
                        plot_confusion_matrix(test_y, pred_y, ['FS', 'OS', 'FE', 'TM'], "Confusion matrix",
                                              args.prdDir + 'predictions_epoch_{}/'.format(str(epoch).zfill(3)) + 'Confusion matrix.jpg', True)
                        # Save the prediction results
                        if args.savePrd:
                            data_smooth(test_label, test_size, test_y, pred_y, pred_dir, args)

                            correct = (test_y == pred_y).sum()
                            accuracy = correct / test_size * 100
                            print('Epoch: {}, After filter Accuracy: {:.1f}%'.format(epoch, accuracy))

                    local_time = time.asctime(time.localtime(time.time()))
                    print('\nTest ends at ' + local_time)
                    plt.close()
                    plt.cla()
                    plt.clf()
                    plt.plot(epoch_list, accuracy_list, 'o-', color='r', label="Accuracy")
                    max_accuracy = max(accuracy_list)
                    max_index = epoch_list[np.argwhere(np.array(accuracy_list) == max_accuracy)[0][0]]

                    plt.ylim(40, 100)
                    plt.xlabel("Epoch. ({}, {:.3f})".format(max_index, max_accuracy))
                    plt.ylabel("Percentage (%)")
                    plt.legend(loc="best")

                    x_major_locator = MultipleLocator(1)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(x_major_locator)

                    plt.savefig(args.prdDir + 'epoch_{}_test_visualization.jpg'.format(str(epo)))
                else:
                    if find_flag == False:
                        print('\n==> No valid checkpoints in directory {}'.format(args.checkpoint))
            else:
                print('\n==> Invalid checkpoint directory: {}'.format(args.checkpoint))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TRCLP Model')

    # Run Encoder
    parser.add_argument('--runEncoder', type=int, default=1, help='Run Encoder')

    # Run Classifier
    parser.add_argument('--runClassifer', type=int, default=1, help='Run Classifier')

    # Only Filter
    parser.add_argument('--onlyFilter', type=int, default=1, help='Only Filter')

    # Load the data augmentation
    parser.add_argument('--augmode', type=int, default=1, help='Augmentation mode')

    # Training for Demo
    parser.add_argument('--demo', type=int, default=1, help='Demo')

    # Test flag
    parser.add_argument('--test', type=int, default=1, help='Test')

    # Encoder epochs
    parser.add_argument('--encoderEpochs', type=int, default=1, help='Set the number of encoder epochs')

    # Classifier epochs
    parser.add_argument('-e', '--classiferEpochs', default=30, type=int, help='Number of total epochs to run (default: 30)')

    # Pretrain flag
    parser.add_argument('--pretrainFlag', default=1, type=int, help='Set the flag to pretrain the model (default: 1)')

    # Train classifier flag
    parser.add_argument('--trainClassifierFlag', default=1, type=int, help='Set the flag to train the classifier (default: 1)')

    # Temperature
    parser.add_argument('--temp', type=float, default=0.07, help='Temperature for loss function')

    # FC size
    parser.add_argument('--FC', default=6, type=int, help='The size of input features (default: 6)')

    # Input size
    parser.add_argument('--inputSize', default=1500, type=int, help='The size of input features (default: 1500)')

    # Eye feature size
    parser.add_argument('--eyeFeatureSize', default=500, type=int, help='The size of eye-in-head features (default: 500)')

    # Head feature size
    parser.add_argument('--headFeatureSize', default=500, type=int, help='The size of head features (default: 500)')

    # Gaze-in-world feature size
    parser.add_argument('--gwFeatureSize', default=500, type=int, help='The size of gaze-in-world features (default: 500)')

    # Number of classes
    parser.add_argument('--numClasses', default=4, type=int, help='The number of classes to predict (default: 4)')

    # Dataset directory
    parser.add_argument('-d', '--datasetDir', default='../../TaskDataset/Cross_User_5_Fold/Test_Fold_1/', type=str,
                        help='The directory that saves the dataset')

    # Train flag
    parser.add_argument('-t', '--trainFlag', default=1, type=int, help='Set the flag to train the model (default: 1)')

    # Path to save checkpoint
    parser.add_argument('-c', '--checkpoint', default='../checkpoint/Cross_User_5_Fold/Test_Fold_1/', type=str,
                        help='Path to save checkpoint')

    # Save prediction flag
    parser.add_argument('--savePrd', default=1, type=int, help='Save the prediction results (1) or not (0) (default: 1)')

    # Prediction directory
    parser.add_argument('-p', '--prdDir', default='../predictions/Cross_User_5_Fold/Test_Fold_1/', type=str,
                        help='The directory that saves the prediction results')

    # Batch size
    parser.add_argument('-b', '--batchSize', default=256, type=int, help='The batch size (default: 256)')

    # Checkpoint interval
    parser.add_argument('-i', '--interval', default=30, type=int, help='The interval that we save the checkpoint (default: 30 epochs)')

    # Initial learning rate
    parser.add_argument('--learningRate', default=1e-2, type=float, help='Initial learning rate (default: 1e-2)')

    # Weight decay
    parser.add_argument('--weightDecay', '--wd', default=1e-4, type=float, help='Weight decay (default: 1e-4)')

    # Gamma for learning rate decay
    parser.add_argument('--gamma', type=float, default=0.75, help='Used to decay learning rate (default: 0.75)')

    # Loss function
    parser.add_argument('--loss', default="CrossEntropy", type=str, help='Loss function to train the network (default: CrossEntropy)')

    # Loss frequency
    parser.add_argument('--lossFrequency', default=3, type=int, help='The frequency that we output the loss in an epoch (default: 3)')

    main(parser.parse_args())
