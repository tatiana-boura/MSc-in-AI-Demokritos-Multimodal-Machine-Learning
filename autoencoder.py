import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import yaml


class VideoClips(Dataset):

    def __init__(self, video_feats, audio_feats, text_feats):

        df_audio = pd.read_csv(audio_feats)
        df_audio.drop_duplicates(subset='song_name', keep='first', inplace=True, ignore_index=True)
        df_text = pd.read_csv(text_feats)
        df_text.drop_duplicates(subset='song_name', keep='first', inplace=True, ignore_index=True)
        df_video = pd.read_csv(video_feats)
        df_video.drop_duplicates(subset='song_name', keep='first', inplace=True, ignore_index=True)

        self.data = pd.merge(df_text, df_audio, on='song_name')
        self.data = pd.merge(self.data, df_video, on='song_name')

        self.data = self.data.drop(['audio_path'], axis=1)
        print(self.data.shape[0])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return torch.tensor(self.data.iloc[index, 1:].values.astype(float), dtype=torch.float32), \
               self.data.loc[index].song_name, index

    def feature_len(self):
        return self.data.shape[1]-1


class Autoencoder(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder1 = nn.Linear(input_dim, latent_dim * 4)
        self.encoder2 = nn.Linear(latent_dim * 4, latent_dim * 2)
        self.encoder3 = nn.Linear(latent_dim * 2, latent_dim)

        self.decoder1 = nn.Linear(latent_dim, latent_dim * 2)
        self.decoder2 = nn.Linear(latent_dim * 2, latent_dim * 4)
        self.decoder3 = nn.Linear(latent_dim * 4, input_dim)

        self.relu = nn.ReLU()

        self.bottleneck = None

    def forward(self, datum):
        x = self.relu(self.encoder1(datum))
        x = self.relu(self.encoder2(x))
        x = self.relu(self.encoder3(x))

        self.bottleneck = x

        x = self.relu(self.decoder1(x))
        x = self.relu(self.decoder2(x))
        x = self.relu(self.decoder3(x))

        return x


def representation_learning():

    with open('./config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    FEATS_PATH = config["FEATS_PATH"]
    CHECKPOINT_PATH = config["CHECKPOINT_PATH"]
    EPOCHS = config["EPOCHS"]

    train_dataset = VideoClips(FEATS_PATH + "\\video_feats.csv", FEATS_PATH + "\\audio_feats.csv",
                               FEATS_PATH + "\\lyrics_embeddings.csv")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    input_dim = train_dataset.feature_len()  # Size of the input data
    latent_dim = 256  # Size of the compressed representation

    embeddings_df = pd.DataFrame(columns=["song_name"] + [str(i) for i in range(latent_dim)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Autoencoder(input_dim, latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for data, _, _ in train_loader:

            inputs = data

            inputs = inputs.view(-1, input_dim).to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss}")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, CHECKPOINT_PATH)

    model.eval()
    target_layer = model.encoder3

    def hook(module, input, o):
        global output
        output = o

    hook_handle = target_layer.register_forward_hook(hook)

    for data, song_name, idx in train_loader:

        inputs = data
        inputs = inputs.view(-1, input_dim).to(device)

        with torch.no_grad():

            model(inputs)
            bottleneck = output  # needs to be -> relu
            print(bottleneck.size())

            bottleneck = bottleneck.tolist()
            idx = idx.tolist()

            for i, s, o in zip(idx, song_name, bottleneck):

                embeddings_df.loc[len(embeddings_df)] = [s] + o

    hook_handle.remove()
    embeddings_df.to_csv("./embeddings.csv", index=False)
