import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import yaml

from sklearn.preprocessing import StandardScaler
from joblib import dump, load


class VideoClips(Dataset):

    def __init__(self, video_feats, audio_feats, text_feats, test=False):

        df_audio = pd.read_csv(audio_feats)
        df_audio.drop_duplicates(subset='song_name', keep='first', inplace=True, ignore_index=True)
        df_text = pd.read_csv(text_feats)
        df_text.drop_duplicates(subset='song_name', keep='first', inplace=True, ignore_index=True)
        df_video = pd.read_csv(video_feats)
        df_video.drop_duplicates(subset='song_name', keep='first', inplace=True, ignore_index=True)

        self.data = pd.merge(df_video, df_audio, on='song_name')
        self.data = pd.merge(self.data, df_text, on='song_name')

        self.data = self.data.drop(['audio_path'], axis=1)

        self.song_names = self.data.iloc[:, 0]

        if not test:
            scaler = StandardScaler()
            scaler.fit(self.data.iloc[:, 1:])
            self.data = scaler.transform(self.data.iloc[:, 1:])

            dump(scaler, 'std_scaler.bin', compress=True)

        else:
            scaler = load('../std_scaler.bin')
            self.data = scaler.transform(self.data.iloc[:, 1:])


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index, :]).float(), self.song_names.loc[index], index

    def feature_len(self):
        return self.data.shape[1]


class Autoencoder(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder0 = nn.Linear(input_dim, 1024)
        self.encoder1 = nn.Linear(1024, 1024)
        self.encoder2 = nn.Linear(1024, 768)
        self.encoder22 = nn.Linear(768, 512)
        self.encoder3 = nn.Linear(512, 256)

        self.decoder0 = nn.Linear(256, 512)
        self.decoder11 = nn.Linear(512, 768)
        self.decoder1 = nn.Linear(768, 1024)
        self.decoder2 = nn.Linear(1024, 1024)
        self.decoder3 = nn.Linear(1024, input_dim)

        self.activation = nn.Tanh()

    def forward(self, datum):
        x = self.activation(self.encoder0(datum))
        x = self.activation(self.encoder1(x))
        x = self.activation(self.encoder2(x))
        x = self.activation(self.encoder22(x))
        x = self.activation(self.encoder3(x))

        x = self.activation(self.decoder0(x))
        x = self.activation(self.decoder11(x))
        x = self.activation(self.decoder1(x))
        x = self.activation(self.decoder2(x))
        x = self.decoder3(x)

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
    # print(model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

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
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss}")

    torch.save(model.state_dict(), CHECKPOINT_PATH)

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
            bottleneck = model.activation(output)
            # print(bottleneck.size())

            bottleneck = bottleneck.tolist()
            idx = idx.tolist()

            for i, s, o in zip(idx, song_name, bottleneck):

                embeddings_df.loc[len(embeddings_df)] = [s] + o

    hook_handle.remove()
    embeddings_df.to_csv("./embeddings.csv", index=False)


def get_representations():

    with open('../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    FEATS_PATH = config["FEATS_PATH"]
    AUTOENC_PATH = config["AUTOENC_PATH"]

    train_dataset = VideoClips(FEATS_PATH + "\\video_feats_test.csv", FEATS_PATH + "\\audio_feats_test.csv",
                                   FEATS_PATH + "\\lyrics_embeddings_test.csv", test=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    input_dim = train_dataset.feature_len()  # Size of the input data
    latent_dim = 256  # Size of the compressed representation

    embeddings_df = pd.DataFrame(columns=["song_name"] + [str(i) for i in range(latent_dim)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Autoencoder(input_dim, latent_dim).to(device)
    model.load_state_dict(torch.load(AUTOENC_PATH))
    model.eval()
    # print(model)

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
            bottleneck = model.activation(output)
            bottleneck = bottleneck.tolist()
            idx = idx.tolist()

            for i, s, o in zip(idx, song_name, bottleneck):

                embeddings_df.loc[len(embeddings_df)] = [s] + o

    hook_handle.remove()
    embeddings_df.to_csv("../embeddings_test.csv", index=False)

