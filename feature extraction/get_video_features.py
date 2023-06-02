import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision import transforms
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device is: {device}')


class VideoDataset(Dataset):
    def __init__(self, video_df, num_of_frames, transform=None):
        self.data = video_df
        self.transform = transform
        self.num_of_frames = num_of_frames

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        video_path = self.data.iloc[idx].path_video

        print(idx, video_path)

        frames = []
        cap = cv2.VideoCapture(video_path)
        count = 0

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(frame_count)
        # interval = frame_count // int(frame_count * 0.2)

        random_idx = random.sample(range(frame_count), self.num_of_frames)

        for i in range(0, frame_count):
            ret, frame = cap.read()

            if not ret:
                break

            if count in random_idx:

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor_frame = self.transform(frame)  
                # tensor_frame.view(3, 256, 256)
                frames.append(tensor_frame)

            count += 1

        cap.release()

        frames_tensor = torch.stack(frames, dim=0)
        frames_tensor = frames_tensor.permute(1, 0, 2, 3)

        return idx, frames_tensor


def get_video_features(data, config):

    if config["MODEL"] == "slow_r50":
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

    model = model.to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {params}\n")

    #print(model.blocks[0])
    #print()
    #print(model.blocks[-1])

    # Define the preprocessing transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create an instance of the VideoDataset
    dataset = VideoDataset(data, num_of_frames=config["NUMBER_OF_FRAMES"], transform=transform)
    dataloader = DataLoader(dataset, batch_size=config["BATCH_SIZE"])

    video_feats_df = pd.DataFrame(columns=[str(i) for i in range(400)])

    model.eval()

    with torch.no_grad():
        '''
        dataloader_iter = iter(dataloader)
        idx, n = next(dataloader_iter)
        print(n.size())
        print(idx)
        output = model(n)
        print(output.size())

        output = output.tolist()
        idx = idx.tolist()

        for i, o in zip(idx, output):
            video_feats_df.loc[i] = o

        print(video_feats_df)

        '''
        for idx, videos in dataloader:

            idx = idx.to(device)
            videos = videos.to(device)

            output = model(videos)
            output = output.tolist()
            idx = idx.tolist()

            for i, o in zip(idx, output):
                video_feats_df.loc[i] = o

    video_feats_df.to_csv("./video_feats.csv", index=False)

    return video_feats_df
