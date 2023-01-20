import argparse
import math
from typing import Dict, List
import torch
import torch.nn as nn
import torch.optim as optim
from models import initialize_model
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import time
import os
import copy
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-dir', '-i', type=str, required=True,
        help='Directory must contain three subdirectories: "train", "val" '
             'and "test" which conform to the pytorch ImageFolder structure.'
    )
    parser.add_argument(
        '--model-name', '-m', type=str, required=True,
        help='Name of the classification model to use. The options are '
             '"resnet", "resnet152", "alexnet", "vgg", "squeezenet", '
             '"densenet", "inception".'
    )
    parser.add_argument(
        '--n-classes', '-n', type=int, required=True,
        help='The number of classes in the training set'
    )
    parser.add_argument(
        '--batch-size', '-b', type=int, default=32,
        help='Training batch size.'
    )
    parser.add_argument(
        '--epochs', '-e', type=int, default=4,
        help='Training batch size.'
    )
    parser.add_argument(
        '--feature-extract', '-f', action='store_true',
        help='If set, freezes all but the last fully connected layer.'
    )
    parser.add_argument(
        '--no-scheduler', action='store_true',
        help='If set, no learning rate scheduler is used.'
    )
    parser.add_argument(
        '--learning-rate', '-e', type=float, default=0.005,
        help='Learning rate. Lower for more precise updates but possibly '
             'slower convergence.'
    )

def get_transforms(input_size: int):
    # Data augmentation and normalization for training
    # Just normalization for validation
    mean_rgb = [0.485, 0.456, 0.406]
    std_rgb = [0.229, 0.224, 0.225]
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean_rgb, std_rgb)
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean_rgb, std_rgb)
        ]),
    }
    return data_transforms

def main(args: argparse.Namespace):
    # Initialize the model for this run
    model_ft, input_size = initialize_model(
        args.model_name, args.n_classes, args.feature_extract, use_pretrained=True
    )

    # Print the model we just instantiated
    print(model_ft)

    # Get augmentations and normalization transforms
    data_transforms = get_transforms(input_size)

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(args.input_dir, x), data_transforms[x]
        ) for x in ['train', 'val']
    }
    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        ) for x in ['train', 'val']
    }

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if args.feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=args.learning_rate, momentum=0.9)

    scheduler = None
    if not args.no_scheduler:
        steps_per_epoch = math.ceil(len(image_datasets['train']) / args.batch_size)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer_ft,
            max_lr=args.learning_rate,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            anneal_strategy='cos'
        )

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(
        model=model_ft,
        device=device,
        dataloaders=dataloaders_dict,
        criterion=criterion,
        optimizer=optimizer_ft,
        num_epochs=args.epochs,
        is_inception=(args.model_name=="inception"),
        scheduler=scheduler
    )

    # Create test dataset and loader
    test_dir = os.path.join(args.input_dir, 'test')
    test_dataset = datasets.ImageFolder(
        test_dir, data_transforms['val']
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # create label mapping (the label with index 2 is actually the string "10" due
    # to alphanumeric order)
    class_to_idx: Dict[str, int] = image_datasets['train'].class_to_idx
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    predictions = predict(test_loader, model_ft)
    predicted_classes = [idx_to_class[pred] for pred in predictions]

    fnames = os.listdir(test_dir)
    fnames = sorted(fnames, key=lambda x: int(x.split('.')[0]))
    csv_path = f'./{args.model_name}.csv'

    df = pd.DataFrame.from_dict({
        'sample': fnames,
        'label': predicted_classes
    })
    df.to_csv(csv_path, index=False)


def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    device,
    num_epochs=25,
    is_inception=False,
    scheduler=None
):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        if scheduler is not None:
                            scheduler.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def predict(loader, model):

    model.eval()

    predictions: List[int] = []

    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.cuda()

            outputs = model(inputs)

            _, pred = torch.max(outputs, 1)

            pred = list(pred.view(-1).cpu().numpy())

            predictions.extend(pred)

    return predictions
