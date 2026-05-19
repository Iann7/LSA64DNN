import torch
import torch.nn as nn
import torch.optim as optim
import constants as const   
from torch.optim.lr_scheduler import ReduceLROnPlateau
#class LSA64Classifier(nn.Module): 
#    def __init__(self, num_classes=64, input_size=const.INPUT_SIZE, hidden_size=64, 
#                 num_layers=2, dropout_p=0.75):
#        super().__init__()
#        self.lstm = nn.GRU(input_size, hidden_size, batch_first=True, 
#                          num_layers=num_layers, dropout=dropout_p, 
#                          bidirectional=True)
#        self.fc = nn.Linear(hidden_size * 2, num_classes)
#
#    def forward(self, x, lengths):
#        # lengths en CPU (del DataLoader)
#        # x en GPU
#        
#        packed = nn.utils.rnn.pack_padded_sequence(
#            x, lengths, batch_first=True, enforce_sorted=False
#        )
#        
#        packed_out, _ = self.lstm(packed)
#        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
#        # out shape: [batch, seq_len, hidden_size*2]
#        
#        # Mover lengths a GPU
#        lengths_gpu = lengths.to(x.device)
#        
#        # Crear máscara con la dimensión correcta
#        batch_size, seq_len, feat_dim = out.shape
#        
#        # mask shape: [batch, seq_len, 1]
#        mask = torch.arange(seq_len, device=x.device).unsqueeze(0) < lengths_gpu.unsqueeze(1)
#        mask = mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
#        
#        # Multiplicar y promediar
#        out_masked = out * mask  # [batch, seq_len, feat_dim]
#        out_sum = out_masked.sum(dim=1)  # [batch, feat_dim]
#        out_mean = out_sum / lengths_gpu.unsqueeze(1).float()  # [batch, feat_dim]
#        
#        return self.fc(out_mean)

class LSA64Classifier(nn.Module): 
    def __init__(self, num_classes=const.NUM_CLASSES, input_size=const.INPUT_SIZE, hidden_size=64, 
                 num_layers=2, dropout_p=0.75):
        super().__init__()
        self.body_parts_lstm = nn.GRU(33*3, hidden_size, batch_first=True, 
                          num_layers=num_layers, dropout=dropout_p, 
                          bidirectional=True)
        self.hands_lstm = nn.GRU(21*3*2, hidden_size, batch_first=True, 
                          num_layers=num_layers, dropout=dropout_p, 
                          bidirectional=True)
        self.fusion = nn.Linear(hidden_size * 4, hidden_size*2)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x, lengths):
        out_body = self._forward_stream(x[:,:,:33*3],self.body_parts_lstm,lengths)
        out_hands = self._forward_stream(x[:,:,33*3:],self.hands_lstm,lengths)
        out_combined = self.fusion(torch.cat([out_body,out_hands],dim=1))
        return self.classifier(out_combined)
    
    def _forward_stream(self,x,gru,lengths):
        # lengths en CPU (del DataLoader)
        # x en GPU
        
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        
        packed_out, _ = gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # out shape: [batch, seq_len, hidden_size*2]
        
        # Mover lengths a GPU
        lengths_gpu = lengths.to(x.device)
        
        # Crear máscara con la dimensión correcta
        batch_size, seq_len, feat_dim = out.shape
        
        # mask shape: [batch, seq_len, 1]
        mask = torch.arange(seq_len, device=x.device).unsqueeze(0) < lengths_gpu.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
        
        # Multiplicar y promediar
        out_masked = out * mask  # [batch, seq_len, feat_dim]
        out_sum = out_masked.sum(dim=1)  # [batch, feat_dim]
        out_mean = out_sum / lengths_gpu.unsqueeze(1).float()  # [batch, feat_dim]
        
        return out_mean