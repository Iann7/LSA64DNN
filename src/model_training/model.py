import torch
import torch.nn as nn
import torch.optim as optim
import constants as const   
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
class BidirectionalCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.hands_to_body_attn = nn.MultiheadAttention(
            hidden_size * 2, num_heads, batch_first=True, dropout=dropout
        )
        self.body_to_hands_attn = nn.MultiheadAttention(
            hidden_size * 2, num_heads, batch_first=True, dropout=dropout
        )
        self.norm_hands = nn.LayerNorm(hidden_size * 2)
        self.norm_body = nn.LayerNorm(hidden_size * 2)
        
    def forward(self, body, hands, mask):
        # 1. Manos atienden a cuerpo (query=hands, key/value=body)
        hands_attended, weights_hand2body = self.hands_to_body_attn(
            query=hands, key=body, value=body,
            key_padding_mask=~mask,
            need_weights=False
        )
        hands_out = self.norm_hands(hands + hands_attended)
        
        # 2. Cuerpo atiende a manos (query=body, key/value=hands)
        body_attended, weights_body2hand = self.body_to_hands_attn(
            query=body, key=hands, value=hands,
            key_padding_mask=~mask,
            need_weights=False
        )
        body_out = self.norm_body(body + body_attended)
        
        return hands_out, body_out, (weights_hand2body, weights_body2hand)
        
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
        self.fusion = nn.Linear(hidden_size * 4, num_classes)
        self.bi_cross_attention = BidirectionalCrossAttention(hidden_size)
    def forward(self, x, lengths):
        out_body = self._forward_stream(x[:,:,:33*3],self.body_parts_lstm,lengths)
        out_hands = self._forward_stream(x[:,:,33*3:],self.hands_lstm,lengths)
        seq_len = out_body.shape[1]
        mask = torch.arange(seq_len, device=x.device).unsqueeze(0) < lengths.to(x.device).unsqueeze(1)
        hands_attended, body_attended, att_weights= self.bi_cross_attention(out_body,out_hands,mask)
        out_body__att_mean = self._masked_mean(body_attended,mask)
        out_hands_att_mean = self._masked_mean(hands_attended,mask)
        out_combined = self.fusion(torch.cat([out_body__att_mean,out_hands_att_mean],dim=1))
        result = self.classifier(out_combined)
        return result
    
    def _forward_stream(self,x,gru,lengths):
        # lengths en CPU (del DataLoader)
        # x en GPU
        lengths_cpu = lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        
        packed_out, _ = gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return out
    
    def _masked_mean(self, x, mask):
        mask_expanded = mask.unsqueeze(-1).float()  
        x_masked = x * mask_expanded
        x_sum = x_masked.sum(dim=1)
        x_count = mask_expanded.sum(dim=1).clamp(min=1) 
        return x_sum / x_count
