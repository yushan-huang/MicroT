import torch
import torchvision
import torchvision.transforms as tr
import pickle
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import r2_score
from torch.utils.data.dataset import random_split
import sys
sys.path.append('./mcunet')
from mcunet.model_zoo import net_id_list, build_model, download_tflite
from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator

class LoadPreDataset(Dataset):
    def __init__(self, pkl_file_path):
        with open(pkl_file_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, teacher_embedding = self.data[idx]
        return image, teacher_embedding

# FIXME: This will need to be adapted to the DINOv2 model
def get_preprocessing():

    preprocess = tr.Compose(
    [
        tr.Resize((224, 224)),  # resized to 70x70
        tr.ToTensor(),
        tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

    return preprocess, preprocess


def get_dataset(train, test):

    train_dataset = torchvision.datasets.ImageNet(
        root="./imagenet", split="train", transform=train
    )

    test_dataset = torchvision.datasets.ImageNet(
        root="./imagenet", split="val", transform=test
    )
    

    return train_dataset, test_dataset


def get_mcunet_student(target_embedding_size: int):

    model, image_size, description = build_model(net_id="mcunet-in3", pretrained=False)  # you can replace net_id with any other option from net_id_list
    model.classifier = torch.nn.Identity()
    model.classifier = torch.nn.Sequential(torch.nn.Linear(160,target_embedding_size))

    return model


def train_student_model(teacher, student, train_loader, test_loader, epochs, device, lr, resume_from=None):
    LR = lr
    student = student.to(device)
    student.train()


    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(student.parameters(), lr=LR, momentum=0.9,weight_decay=5e-4)

    train_losses = []
    test_losses = []
    test_mse_losses = []
    test_rmse_losses = []
    test_r2_scores = []

    start_epoch = 0
    if resume_from is not None:
        print('loading model from:', resume_from)
        checkpoint = torch.load(resume_from)
        student.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    for epoch in range(start_epoch, epochs):
        running_loss = 0.0
        for i, data in enumerate(tqdm(train_loader, 0)):
            inputs, _ = data

            inputs = inputs.to(device)
            with torch.no_grad():
                teacher_outputs = teacher(inputs)

            optimizer.zero_grad()

            student_outputs = student(inputs)

            loss = criterion(student_outputs, teacher_outputs)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        student.eval()
        with torch.no_grad():
            running_test_loss = 0.0
            mse_loss = 0.0
            rmse_loss = 0.0
            r2_score_value = 0.0
            for data in test_loader:
                inputs, _ = data
                inputs = inputs.to(device)

                with torch.no_grad():
                    teacher_outputs = teacher(inputs)

                student_outputs = student(inputs)
                test_loss = criterion(student_outputs, teacher_outputs)
                mse = F.mse_loss(student_outputs, teacher_outputs)
                rmse = torch.sqrt(mse)
                
                running_test_loss += test_loss.item()
                mse_loss += mse.item()
                rmse_loss += rmse.item()
                r2_score_value += r2_score(teacher_outputs.cpu().numpy(), student_outputs.cpu().numpy())

        test_loss = running_test_loss / len(test_loader)
        mse_loss = mse_loss / len(test_loader)
        rmse_loss = rmse_loss / len(test_loader)
        r2_score_value = r2_score_value / len(test_loader)
        test_losses.append(test_loss)
        test_mse_losses.append(mse_loss)
        test_rmse_losses.append(rmse_loss)
        test_r2_scores.append(r2_score_value)
        # Save model after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, f'../imagenet_kd_mcunet_224_epoch_{epoch}_{test_loss}.pth')

        student.train()

        print(f'epoch: {epoch}, train loss: {train_loss}, test loss: {test_loss}, test MSE: {mse_loss}, test RMSE: {rmse_loss}, test R^2: {r2_score_value}',optimizer.param_groups[0]['lr'])
        
    
    # Save student model
    torch.save(student.state_dict(), '../imagenet_kd_mcunet_224.pth')
    
    return student


def main():
    device = torch.device("cuda")
    preprocessing_train, preprocessing_test = get_preprocessing()
    train_dataset_ori, test_dataset_ori = get_dataset(preprocessing_train, preprocessing_test)

    # We load the DINOv2 model from Facebook Research: Smallest version with 21M parameters
    teacher = torch.hub.load('./dinov2/', 'dinov2_vits14',source='local').cuda() # Note: Revise the path based on the environment
    # Freeze all layers in the base_model
    for param in teacher.parameters():
        param.requires_grad = False
    # Teacher head is expected to be (s: 384, g: 1536)
    teacher_embedding_size = 384

    # student_1 = get_mobilenet_v3_student(target_embedding_size=teacher_embedding_size)
    student_1 = get_mcunet_student(target_embedding_size=teacher_embedding_size)


    # Move models to the device
    # DINOv2 seem to expect the model and data to be on GPU.
    teacher = teacher.to(device)
    student_1 = student_1.to(device)

    # test the consistency of the size of teacher embedding and student embedding
    first_image = train_dataset_ori[0][0].to(device)

    teacher.eval()
    student_1.eval()

    with torch.no_grad():
        teacher_embedding = teacher(first_image.unsqueeze(0))
        student_embedding = student_1(first_image.unsqueeze(0))

        check_msg = "Embedding shapes do not match. {} vs {}"
        check_msg = check_msg.format(teacher_embedding.shape, student_embedding.shape)
        assert teacher_embedding.shape == student_embedding.shape, check_msg
    

    train_loader = DataLoader(train_dataset_ori, batch_size=128, shuffle=True, num_workers=25)
    test_loader = DataLoader(test_dataset_ori, batch_size=128, shuffle=False, num_workers=25)

    # Train the student model
    epochs = 50 # Define the number of epochs
    LR = 0.01
    student_1 = train_student_model(teacher,student_1, train_loader, test_loader, epochs, device, LR)


if __name__ == "__main__":
    main()