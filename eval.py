import torch
import torchvision
import torchvision.transforms as tr
from torchvision import datasets
import numpy as np
import torch.nn as nn
from torch.utils.data import  DataLoader
from tqdm import tqdm
from torch.utils.data.dataset import random_split
from sklearn.metrics import accuracy_score
from mcunet.model_zoo import net_id_list, build_model, download_tflite
import joblib


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_lr_classifier(model_path):
    model = joblib.load(model_path)
    return model



class PartialModel(nn.Module):
    def __init__(self, original_model, cut_point=9, num_features=40, num_classes=160):
        super(PartialModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            original_model.first_conv, 
            *original_model.blocks[0:cut_point]  # please revise the number to determin the optimal segmentation point
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x):
        features = self.feature_extractor(x)
        pooled_features = self.global_avg_pool(features)
        flattened_features = pooled_features.view(pooled_features.size(0), -1)
        return flattened_features


def get_mcunet_student(model_path):

    model, image_size, description = build_model(net_id="mcunet-in3", pretrained=True)  # you can replace net_id with any other option from net_id_list
    model.classifier = torch.nn.Identity()
    model.classifier = torch.nn.Sequential(torch.nn.Linear(160,384))
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.classifier = nn.Identity()

    return model


def get_dataset(dataset_name):

    # data preprocessing
    image_resolution = 224
    
    if dataset_name == 'pet':
        preprocess = tr.Compose([
            tr.Resize((image_resolution, image_resolution)),
            tr.ToTensor(),
            tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset_path = "./Oxford_Pet_Dataset"
        dataset = datasets.ImageFolder(root=dataset_path, transform=preprocess)

        train_size = int(0.7 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        test_dataset_indices = torch.load('/./test_dataset_indices_pet_MCUNet_224.pth')
        test_dataset = torch.utils.data.Subset(dataset, test_dataset_indices)


        trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=25)
        testloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=25)

    
    elif dataset_name == 'plant':
        preprocess = tr.Compose([
            tr.Resize((image_resolution, image_resolution)),
            tr.ToTensor(),
            tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset_path = "./PlantCLEF_Subset"

        dataset = datasets.ImageFolder(root=dataset_path, transform=preprocess)

        train_size = int(0.7 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        test_dataset_indices = torch.load('./test_dataset_indices_plant_MCUNet_224.pth')
        test_dataset = torch.utils.data.Subset(dataset, test_dataset_indices)

        trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=25)
        testloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=25)
    
    elif dataset_name == 'bird':

        parent_folder = '/home/yushan/battery-free/bird' 


        preprocess = tr.Compose([
            tr.Resize((image_resolution, image_resolution)),
            tr.ToTensor(),
            tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        full_dataset = datasets.ImageFolder(parent_folder, transform=preprocess)

        train_size = int(0.7 * len(full_dataset))
        test_size = len(full_dataset) - train_size

        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
        test_dataset_indices = torch.load('./test_dataset_indices_bird_MCUNet_224.pth')
        test_dataset = torch.utils.data.Subset(full_dataset, test_dataset_indices)

        trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=25)
        testloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=25)


    return trainloader, testloader



def get_features(loader, full_model, part_model):
    full_model.eval()
    part_model.eval()
    features_full = []
    features_part = []
    labels = []

    with torch.no_grad():
        for inputs, targets in tqdm(loader):
            inputs = inputs.cuda()
            outputs_full = full_model(inputs) 
            outputs_part = part_model(inputs)
            # Collect features and labels
            features_full.append(outputs_full.cpu())
            features_part.append(outputs_part.cpu())
            labels.append(targets)

    # Convert the list of tensors to a single tensor
    features_full = torch.cat(features_full, dim=0)
    features_part = torch.cat(features_part, dim=0)
    labels = torch.cat(labels, dim=0)

    return features_full, features_part, labels



def test_single_LR(dataset,model_path, full_classifier_path, part_classifier_path, confidence_threshold):
    print('loading dataset')
    if dataset == 'pet':
        print('Dataset pet')
        num_class = 37
        neighbour_range = 19
    elif dataset == 'plant':
        print('Dataset plant')
        num_class = 20
        neighbour_range = 10
    elif dataset == 'bird':
        print('Dataset bird')
        num_class = 200
        neighbour_range = 100
    

    else:
        raise ValueError("Unknown dataset:{}".format(dataset))
    trainloader, testloader = get_dataset(dataset)


    full_model = get_mcunet_student(model_path).cuda()
    part_model = PartialModel(full_model)
    test_features_full, test_features_part, test_labels = get_features(testloader, full_model, part_model)

    full_nn_classifier = load_lr_classifier(model_path=full_classifier_path)
    part_nn_classifier = load_lr_classifier(model_path=part_classifier_path)

    test_features_full = test_features_full
    test_features_part = test_features_part
    test_labels = test_labels

    test_features_full = test_features_full.cpu().detach().numpy()
    test_features_part = test_features_part.cpu().detach().numpy()
    test_labels = test_labels.cpu().detach().numpy()


    probabilities_part = part_nn_classifier.predict_proba(test_features_part)
    confidence_part = np.max(probabilities_part, axis=1)
    predicted_labels_part = np.argmax(probabilities_part, axis=1)

    mask = confidence_part < confidence_threshold


    part_model_used = np.sum(~mask)
    full_model_used = np.sum(mask)

    if full_model_used > 0:
        probabilities_full = full_nn_classifier.predict_proba(test_features_full[mask])
        predicted_labels_full = np.argmax(probabilities_full, axis=1)
        
        # update results
        predicted_labels_part[mask] = predicted_labels_full

    # calculate final acc
    final_accuracy = accuracy_score(test_labels, predicted_labels_part)

    print(f'Final Joint Model Accuracy LR: {final_accuracy*100}, Rate: {part_model_used/(part_model_used+ full_model_used)}')




def main():
    # single test
    model_path = './imagenet_kd_mcunet_224.pth'


    part_classifier_path_LR = './main_classifier/LR_model_part_MCUNet_224.pkl' # the path of trained classifier
    full_classifier_path_LR = './main_classifier/LR_model_full_MCUNet_224.pkl'

    confidence_threshhold_LR = 0.5

    test_single_LR('pet',model_path, full_classifier_path_LR, part_classifier_path_LR, confidence_threshhold_LR)





if __name__ == "__main__":
    main()
