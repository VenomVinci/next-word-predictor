import torch
import torch.nn as nn
import torch.nn.functional as F

# Example LSTM model class (adapt to your exact model)
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

def load_model(device):
    # Load your trained model checkpoint
    checkpoint_path = "sherlock_model.pth"  # Adjust path
    vocab = torch.load("vocab.pth")  # Your vocab dictionary

    model = LSTMModel(vocab_size=len(vocab), embed_size=128, hidden_size=256, num_layers=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model, vocab

def predict_next_word(model, vocab, text, top_k=5, device="cpu"):
    """
    Given input text, predict top_k next words.
    """
    tokens = [vocab.get(token, 0) for token in text.lower().split()]
    input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    
    with torch.no_grad():
        output, _ = model(input_tensor)
        probs = F.softmax(output, dim=-1)
        top_probs, top_idx = torch.topk(probs, top_k)
    
    idx_to_word = {v:k for k,v in vocab.items()}
    return [idx_to_word[idx.item()] for idx in top_idx[0]]
