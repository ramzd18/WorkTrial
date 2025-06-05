import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from inference.data_loader import get_dataloaders
from tqdm import tqdm

class SpectrogramCNN(nn.Module):
    """2D CNN for processing spectrograms"""
    
    def __init__(self):
        super(SpectrogramCNN, self).__init__()
        
        self.spectrogram = T.Spectrogram(
            n_fft=1024,
            hop_length=512,
            power=2.0
        )
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.conv4 = nn.Conv2d(128, 512, kernel_size=(3, 3), stride=(2, 2), padding=1)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.dropout = nn.Dropout(0.2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = self.spectrogram(x) 
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)
        
        x = self.global_pool(x)  
        x = x.view(x.size(0), -1) 
        return x
    
class TemporalLSTM(nn.Module):
    """LSTM for capturing temporal patterns"""
    
    def __init__(self, input_size, hidden_size=512, num_layers=5):
        super(TemporalLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.output_size = hidden_size * 2 
        
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        output = lstm_out[:, -1, :]  
        forward_h = hidden[-2]    
        backward_h = hidden[-1]           
        final_h = torch.cat((forward_h, backward_h), dim=1)
        # print("FINAL H SHAPE", final_h.shape)
        return final_h
    
class MultiModalMOSPredictor(nn.Module):
    """Complete MOS prediction model combining multiple modalities"""
    
    def __init__(self, feature_dim, num_classes=1):
        super(MultiModalMOSPredictor, self).__init__()
        
        self.spectrogram_cnn = SpectrogramCNN()
        
        self.chunk_size = 8000  
        self.temporal_lstm = TemporalLSTM(input_size=512, hidden_size=512)
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        fusion_input_size = self.temporal_lstm.output_size + 64
        # print("FUSION INPUT SIZE", fusion_input_size)

        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        self.output_activation = nn.Sigmoid()
        
    def forward(self, audio, features):
        audio = audio.squeeze(1)
        batch_size = audio.size(0)
        audio_length = audio.size(1)
        # print("AUDIO LENGTH", audio.shape)
        processed_features = self.feature_processor(features)
        # print("PROCESSED FEATURES SHAPE", processed_features.shape)
        chunks = []
        num_chunks = audio_length // self.chunk_size
        # No chunking: process the whole audio in one shot with the CNN
        spec_features = self.spectrogram_cnn(audio)
        # print("SPEC FEATURES SHAPE", spec_features.shape)
        temporal_input = spec_features.unsqueeze(1)  # Add sequence dimension for LSTM compatibility
        temporal_features = self.temporal_lstm(temporal_input)
        # print("TEMPORAL FEATURES SHAPE", temporal_features.shape)
        combined_features = torch.cat([temporal_features, processed_features], dim=1)
        # print("COMBINED FEATURES SHAPE", combined_features.shape)
        output = self.fusion_network(combined_features)
        # print("FINSIHED W THAT")
        output = self.output_activation(output) * 4 + 1
        
        return output
    
class MOSTrainer:
    """Training utilities for the MOS prediction model"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5
        )
        
    def train_epoch(self, batch):
        """Process a single batch during training"""
        audio = batch['audio'].to(self.device)
        features = batch['features'].to(self.device)
        mos = batch['mos_score'].to(self.device)
        
        self.optimizer.zero_grad()
        
        predictions = self.model(audio, features)
        loss = self.criterion(predictions, mos)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, batch):
        """Process a single batch during validation"""
        audio = batch['audio'].to(self.device)
        features = batch['features'].to(self.device)
        mos = batch['mos_score'].to(self.device)
        
        predictions = self.model(audio, features)
        loss = self.criterion(predictions, mos)
        
        return loss.item()
    
    def train(self, train_loader, val_loader, num_epochs=30):
        """Main training loop with progress bars"""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            for batch in train_pbar:
                loss = self.train_epoch(batch)
                train_loss += loss
                train_pbar.set_postfix({'loss': f'{loss:.4f}'})
            train_loss /= len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            with torch.no_grad():
                for batch in val_pbar:
                    loss = self.validate(batch)
                    val_loss += loss
                    val_pbar.set_postfix({'loss': f'{loss:.4f}'})
            val_loss /= len(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Print epoch summary
            print(f'\nEpoch {epoch+1}/{num_epochs} Summary:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'mos_deep_network.pth')
                print(f'  New best model saved!')

def main():
    model = MultiModalMOSPredictor(feature_dim=64)
    trainer = MOSTrainer(model)
    train_loader, val_loader = get_dataloaders()
    trainer.train(train_loader, val_loader, num_epochs=30)

if __name__ == "__main__":
    main()
    print("DONE")