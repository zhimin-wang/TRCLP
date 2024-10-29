"""Module for loading and preprocessing datasets, including data augmentation, for the model."""

import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
from sklearn.metrics import confusion_matrix

from .augmentations import dataAugmentation, only_scale


class MyDataset_only_aug(data.Dataset):
    """Dataset class for applying data augmentation during training."""

    def __init__(self, features, labels, training):
        """Initializes the dataset.

        Args:
            features: Input features.
            labels: Corresponding labels.
            training: Boolean indicating if the dataset is for training.
        """
        self.features = features
        self.labels = labels
        self.training = training

    def __getitem__(self, index):
        """Retrieves a single data point and applies augmentation if in training mode.

        Args:
            index: Index of the data point.

        Returns:
            Tuple of augmented feature and target label.
        """
        feature = self.features[index]
        target = self.labels[index]
        if self.training:
            if target[0] == 0:
                # Apply basic data augmentation
                aug_data = dataAugmentation(feature)
            else:
                # Apply stronger data augmentation
                aug_data = only_scale(feature)
            return aug_data, target[1:]
        else:
            return feature, target

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.features)


class MyDataset_Contrast_aug(data.Dataset):
    """Dataset class for contrastive data augmentation."""

    def __init__(self, features, labels, trainClassifierFlag, training):
        """Initializes the dataset.

        Args:
            features: Input features.
            labels: Corresponding labels.
            trainClassifierFlag: Flag indicating if training the classifier.
            training: Boolean indicating if the dataset is for training.
        """
        self.features = features
        self.labels = labels
        self.training = training
        self.trainClassifierFlag = trainClassifierFlag

    def __getitem__(self, index):
        """Retrieves a single data point and applies augmentation if in training mode.

        Args:
            index: Index of the data point.

        Returns:
            Tuple of original feature, augmented feature, and target label.
        """
        feature = self.features[index]
        target = self.labels[index]
        if self.training:
            if target[0] == 0:
                aug_data = dataAugmentation(feature)
            else:
                aug_data = only_scale(feature)
            return feature, aug_data, target[1:]
        else:
            return feature, target

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.features)


class MyDataset(data.Dataset):
    """Standard dataset class without augmentation."""

    def __init__(self, features, labels, training):
        """Initializes the dataset.

        Args:
            features: Input features.
            labels: Corresponding labels.
            training: Boolean indicating if the dataset is for training.
        """
        self.features = features
        self.labels = labels
        self.training = training

    def __getitem__(self, index):
        """Retrieves a single data point.

        Args:
            index: Index of the data point.

        Returns:
            Tuple of feature and target label.
        """
        feature = self.features[index]
        target = self.labels[index]
        if self.training:
            return feature, target[1:]
        else:
            return feature, target

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.features)


training_mean = []
training_std = []


def Load_20_P_allY_TestData(strip, datasetDir, batch_size):
    """Loads the test data.

    Args:
        strip: Number of columns to strip from labels.
        datasetDir: Directory of the dataset.
        batch_size: Batch size for DataLoader.

    Returns:
        test_loader: DataLoader for the test dataset.
        label_testY: Original test labels.
    """
    print("\nLoading the test data...")
    testX = torch.from_numpy(np.load(os.path.join(datasetDir, 'DataX.npy'))).float()
    print('\nTest Data Size: {}'.format(list(testX.size())))
    label_testY = np.load(os.path.join(datasetDir, 'DataY.npy'), allow_pickle=True)
    testY = torch.from_numpy(label_testY[:, strip:].astype(np.int32)).long()

    test_dataset = MyDataset(testX, testY, training=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return test_loader, label_testY


def Load_20_P_allY_TrainingData(strip, datasetDir, batch_size, augMode, trainClassifierFlag):
    """Loads and preprocesses training data from multiple datasets.

    Args:
        strip: Number of columns to strip from labels.
        datasetDir: Directory of the current dataset.
        batch_size: Batch size for DataLoader.
        augMode: Augmentation mode.
        trainClassifierFlag: Flag indicating whether training the classifier.

    Returns:
        train_loader: DataLoader for the training dataset.
    """
    c_datasetDir = str(datasetDir)
    filePath = os.path.join(c_datasetDir, "../")
    fileList = os.listdir(filePath)

    print("\nLoading the training data...")
    trainingX = []
    trainingY = []
    for fileName in fileList:
        if fileName != os.path.basename(os.path.normpath(c_datasetDir)):
            print("fileName: ", fileName)
            dataPath = os.path.join(filePath, fileName)
            dataX = np.load(os.path.join(dataPath, 'DataX.npy'))
            label_trainingY = np.load(os.path.join(dataPath, 'DataY.npy'), allow_pickle=True)
            if len(trainingX) == 0:
                trainingX = np.array(dataX)
                trainingY = label_trainingY[:, strip:]
            else:
                trainingX = np.vstack((trainingX, dataX))
                trainingY = np.vstack((trainingY, label_trainingY[:, strip:]))

    trainingX = torch.from_numpy(trainingX).float()
    print('\nTraining Data Size: {}'.format(list(trainingX.size())))

    trainingY = torch.from_numpy(trainingY.astype(np.int32)).long()
    print('\nTraining Data Size: {}'.format(list(trainingY.size())))

    if augMode == 0:  # Without augmentation
        train_dataset = MyDataset(trainingX, trainingY, training=True)
    elif augMode == 1:  # Basic augmentation
        train_dataset = MyDataset_only_aug(trainingX, trainingY, training=True)
    elif augMode == 2:  # Contrastive learning
        train_dataset = MyDataset_Contrast_aug(trainingX, trainingY, trainClassifierFlag, training=True)

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader


def Load_20_P_allY_TrainingDataForDemo(strip, datasetDir, batch_size):
    """Loads training data for demo purposes from all datasets.

    Args:
        strip: Number of columns to strip from labels.
        datasetDir: Directory of the current dataset.
        batch_size: Batch size for DataLoader.

    Returns:
        train_loader: DataLoader for the training dataset.
    """
    c_datasetDir = str(datasetDir)
    filePath = os.path.join(c_datasetDir, "../")
    fileList = os.listdir(filePath)

    print("\nLoading the training data...")
    trainingX = []
    trainingY = []
    for fileName in fileList:
        print("fileName: ", fileName)
        dataPath = os.path.join(filePath, fileName)
        dataX = np.load(os.path.join(dataPath, 'DataX.npy'))
        label_trainingY = np.load(os.path.join(dataPath, 'DataY.npy'), allow_pickle=True)
        if len(trainingX) == 0:
            trainingX = np.array(dataX)
            trainingY = label_trainingY[:, strip:]
        else:
            trainingX = np.vstack((trainingX, dataX))
            trainingY = np.vstack((trainingY, label_trainingY[:, strip:]))

    trainingX = torch.from_numpy(trainingX).float()
    print('\nTraining Data Size: {}'.format(list(trainingX.size())))

    trainingY = torch.from_numpy(trainingY.astype(np.int32)).long()
    print('\nTraining Data Size: {}'.format(list(trainingY.size())))

    train_dataset = MyDataset_Contrast_aug(trainingX, trainingY, trainClassifierFlag=False, training=True)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader


def LoadTrainingData(datasetDir, batch_size):
    """Loads the training data from the specified dataset.

    Args:
        datasetDir: Directory of the dataset.
        batch_size: Batch size for DataLoader.

    Returns:
        train_loader: DataLoader for the training dataset.
    """
    print("\nLoading the training data...")
    trainingX = torch.from_numpy(np.load(os.path.join(datasetDir, 'trainingX.npy'))).float()
    print('\nTraining Data Size: {}'.format(list(trainingX.size())))
    label_trainingY = np.load(os.path.join(datasetDir, 'trainingY.npy'), allow_pickle=True)
    trainingY = torch.from_numpy(label_trainingY[:, 3].astype(np.int32)).long()

    train_dataset = MyDataset(trainingX, trainingY, training=True)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader


def Load_20_P_allY_TestData_RF(strip, datasetDir, batch_size, eyeFeatureSize):
    """Loads test data for Random Forest model.

    Args:
        strip: Number of columns to strip from labels.
        datasetDir: Directory of the dataset.
        batch_size: Batch size (unused in this function).
        eyeFeatureSize: Size of eye feature.

    Returns:
        testX: Test features.
        testY: Test labels.
    """
    print("\nLoading the test data...")
    testX = np.load(os.path.join(datasetDir, 'DataX.npy'))
    print('\nTest Data Size: {}'.format(testX.shape))
    label_testY = np.load(os.path.join(datasetDir, 'DataY.npy'), allow_pickle=True)
    testY = label_testY[:, strip:].astype(np.int32)
    feature_len = int(eyeFeatureSize / 2)
    return testX, testY[:, feature_len - 1:feature_len].reshape(-1).astype(np.int32)


def Load_20_P_allY_TrainingData_RF(strip, datasetDir, batch_size, augMode, trainClassifierFlag, eyeFeatureSize):
    """Loads training data for Random Forest model.

    Args:
        strip: Number of columns to strip from labels.
        datasetDir: Directory of the current dataset.
        batch_size: Batch size (unused in this function).
        augMode: Augmentation mode (unused in this function).
        trainClassifierFlag: Flag indicating whether training the classifier (unused).
        eyeFeatureSize: Size of eye feature.

    Returns:
        trainingX: Training features.
        trainingY: Training labels.
    """
    c_datasetDir = str(datasetDir)
    filePath = os.path.join(c_datasetDir, "../")
    fileList = os.listdir(filePath)

    print("\nLoading the training data...")
    trainingX = []
    trainingY = []
    for fileName in fileList:
        if fileName != os.path.basename(os.path.normpath(c_datasetDir)):
            print("fileName: ", fileName)
            dataPath = os.path.join(filePath, fileName)
            dataX = np.load(os.path.join(dataPath, 'DataX.npy'))
            label_trainingY = np.load(os.path.join(dataPath, 'DataY.npy'), allow_pickle=True)
            if len(trainingX) == 0:
                trainingX = np.array(dataX)
                trainingY = label_trainingY[:, strip:]
            else:
                trainingX = np.vstack((trainingX, dataX))
                trainingY = np.vstack((trainingY, label_trainingY[:, strip:]))

    print('\nTraining Data Size: {}'.format(trainingX.shape))
    feature_len = int(eyeFeatureSize / 2)
    return trainingX, trainingY[:, feature_len - 1:feature_len].reshape(-1).astype(np.int32)


def LoadTestData(datasetDir, batch_size):
    """Loads the test data from the specified dataset.

    Args:
        datasetDir: Directory of the dataset.
        batch_size: Batch size for DataLoader.

    Returns:
        test_loader: DataLoader for the test dataset.
        label_testY: Original test labels.
    """
    print("\nLoading the test data...")
    testX = torch.from_numpy(np.load(os.path.join(datasetDir, 'testX.npy'))).float()
    print('\nTest Data Size: {}'.format(list(testX.size())))
    label_testY = np.load(os.path.join(datasetDir, 'testY.npy'), allow_pickle=True)
    testY = torch.from_numpy(label_testY[:, 3].astype(np.int32)).long()
    test_dataset = MyDataset(testX, testY, training=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return test_loader, label_testY


def plot_confusion_matrix(test_y, pred_y, classes, title, path, normalize=False, cmap=plt.cm.Blues):
    """Plots the confusion matrix.

    Args:
        test_y: True labels.
        pred_y: Predicted labels.
        classes: List of class names.
        title: Title of the plot.
        path: Path to save the plot.
        normalize: If True, normalize the confusion matrix.
        cmap: Color map for the plot.
    """
    cm = confusion_matrix(test_y, pred_y)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    else:
        print('Confusion matrix, without normalization:')
        print(cm)

    plt.close()
    plt.cla()
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # Ensure all data is displayed
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.tight_layout()
    plt.ylabel('True label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted label', fontsize=12, fontweight='bold')
    plt.savefig(path)
