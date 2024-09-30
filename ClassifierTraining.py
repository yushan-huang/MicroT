import torch
import torchvision
import torchvision.transforms as tr
from torchvision import datasets
import torch.nn as nn
from torch.utils.data import  DataLoader
from tqdm import tqdm
from torch.utils.data.dataset import random_split
import sys
sys.path.append('/home/yushan/battery-free/mcunet')
from mcunet.model_zoo import net_id_list, build_model, download_tflite
from classifier_train import LR_classifier



device = 'cuda' if torch.cuda.is_available() else 'cpu'


# If use the Part Model.
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

    return model


# FIXME: Placeholde that loads the CIFAR10 dataset
def get_dataset(dataset_name) -> torch.utils.data.Dataset:

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

        trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=25)
        testloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=25)
    
    elif dataset_name == 'bird':

        parent_folder = './bird' 


        preprocess = tr.Compose([
            tr.Resize((image_resolution, image_resolution)),
            tr.ToTensor(),
            tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        full_dataset = datasets.ImageFolder(parent_folder, transform=preprocess)

        train_size = int(0.7 * len(full_dataset))
        test_size = len(full_dataset) - train_size

        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

        trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=25)
        testloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=25)


    return trainloader, testloader


# get embedding features from the teacher
def get_features_from_teacher(loader, teacher):
    teacher.eval()
    features = []
    labels = []

    with torch.no_grad():
        for inputs, targets in tqdm(loader):
            inputs = inputs.cuda()
            outputs = teacher(inputs) 
            features.extend(outputs.cpu().detach().numpy())
            labels.extend(targets.numpy())
    
    return features, labels


def test_single(dataset,model_path):
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
    # get teacher embedding features
    # load model
    teacher = get_mcunet_student(model_path).cuda()
    teacher.cuda()
    train_features, train_labels = get_features_from_teacher(trainloader, teacher)
    test_features, test_labels = get_features_from_teacher(testloader, teacher)
    print('loading finished')

    LR_acc,_ = LR_classifier(train_features, train_labels, test_features, test_labels)

    print('LR acc:', LR_acc)

def main():
    model_path = './imagenet_kd_mcunet.pth'
    test_single('plant',model_path)


if __name__ == "__main__":
    main()
