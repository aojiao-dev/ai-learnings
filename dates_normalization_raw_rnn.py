import csv
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# --------------------------------------------------------
# Config
# --------------------------------------------------------

train_file = "data/DATES/messy_dates_train.csv"
eval_file = "data/DATES/messy_dates_eval.csv"
test_file = "data/DATES/messy_dates_test.csv"

SEED = 42
BATCH_SIZE = 128
EPOCHS = 10
LR = 2e-3

EMBED_DIM = 128
HIDDEN_DIM = 512
MAX_LEN_IN = 16
MAX_LEN_OUT = 16
TEACHER_FORCING = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)

random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# --------------------------------------------------------
# Verify Training Data
# --------------------------------------------------------

training_set = []
eval_set = []
test_set = []


def load_file(file: str, set: dict):
    with open(file, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            assert len(row) == 2
            assert row[0] and row[1]
            set.append(row)


load_file(train_file, training_set)
load_file(eval_file, eval_set)
load_file(test_file, test_set)

print("training_set: ", len(training_set))
print("eval_set: ", len(eval_set))
print("test_set: ", len(test_set))

# --------------------------------------------------------
# Vocab (character-level)
# --------------------------------------------------------
# Special tokens
PAD = "<pad>"
SOS = "<s>"
EOS = "</s>"


def build_vocab(
    pairs: List[Tuple[str, str]], extra_pairs: List[Tuple[str, str]]
) -> Dict[str, int]:
    chars = set()
    for key, value in pairs + extra_pairs:
        chars.update(list(key))
        chars.update(list(value))
    itos = [PAD, SOS, EOS] + sorted(chars)
    stoi = {ch: i for i, ch in enumerate(itos)}
    return {"itos": itos, "stoi": stoi}


vocab = build_vocab(training_set, eval_set + test_set)
ITOS = vocab["itos"]
STOI = vocab["stoi"]
PAD_IDX = STOI[PAD]
SOS_IDX = STOI[SOS]
EOS_IDX = STOI[EOS]
VOCAB_SIZE = len(ITOS)
print("vocab size:", VOCAB_SIZE)


# --------------------------------------------------------
# Construct Data Loader
# --------------------------------------------------------
def text_to_idx_list(s: str, stoi: Dict[str, int], max_len: int) -> List[int]:
    s = s[:max_len]
    return [stoi[ch] for ch in s]


@dataclass
class DatePair:
    src: str  # messy dates
    target: str  # canonical


class DatesDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]]):
        self.examples = [DatePair(src=p[0], target=p[1]) for p in pairs]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        src_ids = text_to_idx_list(example.src, STOI, MAX_LEN_IN)[::-1]
        target_ids = text_to_idx_list(example.target, STOI, MAX_LEN_OUT - 1)
        target_ids = target_ids + [EOS_IDX]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(
            target_ids, dtype=torch.long
        )


def pad(seqs: List[torch.Tensor], pad_idx: int) -> torch.Tensor:
    max_len = max(len(x) for x in seqs)
    out = torch.full((len(seqs), max_len), pad_idx, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
    return out


def collate_fn(batch):
    src_list, target_list = zip(*batch)
    src_lens = torch.tensor([len(s) for s in src_list], dtype=torch.long)
    src_pad = pad(list(src_list), PAD_IDX)

    decoder_input_list, target_out_list = [], []
    for t in target_list:
        decoder_input = torch.cat([torch.tensor([SOS_IDX]), t[:-1]])
        decoder_input_list.append(decoder_input)
        target_out_list.append(t)

    dec_in_pad = pad(decoder_input_list, PAD_IDX)
    tgt_pad = pad(target_out_list, PAD_IDX)

    return src_pad, src_lens, dec_in_pad, tgt_pad


train_ds = DatesDataset(training_set)
eval_ds = DatesDataset(eval_set)
test_ds = DatesDataset(test_set)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
eval_loader = DataLoader(
    eval_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)


# --------------------------------------------------------
# Seq2Seq model with vanilla RNN (no attention)
# --------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=PAD_IDX
        )
        self.rnn = nn.RNN(
            input_size=embed_dim, hidden_size=hidden_dim, batch_first=True
        )

    def forward(self, src, src_len):
        x = self.embed(src)  # [B, T, E]
        packed = pack_padded_sequence(
            x, src_len.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs_packed, hidden = self.rnn(packed)  # hidden: [1, B, H]
        outputs, _ = pad_packed_sequence(outputs_packed, batch_first=True)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=PAD_IDX
        )
        self.rnn = nn.RNN(
            input_size=embed_dim, hidden_size=hidden_dim, batch_first=True
        )
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, decoder_input, hidden):
        x = self.embed(decoder_input)  # [B, T, E]
        outputs, hidden = self.rnn(x, hidden)  # outputs: [B, T, H]
        logits = self.proj(outputs)  # [B, T, V]
        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_dim, hidden_dim)
        self.decoder = Decoder(vocab_size, embed_dim, hidden_dim)

    def forward(self, src, src_len, decoder_input, teacher_forcing: float = 1.0):
        B, T_dec = decoder_input.size(0), decoder_input.size(1)
        V = self.decoder.proj.out_features
        all_logits = torch.zeros(B, T_dec, V, device=src.device)

        _, hidden = self.encoder(src, src_len)
        step_input = decoder_input[:, 0].unsqueeze(1)  # [B, 1]

        for t in range(T_dec):
            logits_step, hidden = self.decoder(step_input, hidden)  # [B, 1, V]
            all_logits[:, t, :] = logits_step.squeeze(1)
            use_tf = (t < T_dec - 1) and (random.random() < teacher_forcing)
            if use_tf:
                step_input = decoder_input[:, t + 1].unsqueeze(1)
            else:
                step_input = logits_step.argmax(dim=2).detach()  # [B, 1]

        return all_logits

    def inference(self, src, src_len, max_len=MAX_LEN_OUT):
        B = src.size(0)
        _, hidden = self.encoder(src, src_len)
        step_input = torch.full((B, 1), SOS_IDX, dtype=torch.long, device=src.device)
        outputs = []

        for _ in range(max_len):
            logits, hidden = self.decoder(step_input, hidden)  # [B, 1, V]
            step_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
            outputs.append(step_tokens)
            step_input = step_tokens
            # early stop if all EOS
            if (step_tokens == EOS_IDX).all():
                break

        return torch.cat(outputs, dim=1)  # [B, T_out]


model = Seq2Seq(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM).to(DEVICE)

# --------------------------------------------------------
# Loss / Optimization
# --------------------------------------------------------
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # not AdamW


# --------------------------------------------------------
# Train / Eval loops
# --------------------------------------------------------
def _trim_decoder(dec_in, tgt):
    dec_lens = (tgt != PAD_IDX).sum(dim=1)  # includes EOS
    max_dec_len = int(dec_lens.max().item())
    return dec_in[:, :max_dec_len], tgt[:, :max_dec_len]


def run_eval(loader):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.inference_mode():
        for src, src_len, decoder_input, target in loader:
            src = src.to(DEVICE)
            src_len = src_len.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            target = target.to(DEVICE)

            dec_in_trim, tgt_trim = _trim_decoder(decoder_input, target)
            logits = model(src, src_len, dec_in_trim, teacher_forcing=0.0)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_trim.reshape(-1))

            n_tokens = (tgt_trim != PAD_IDX).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    return total_loss / max(1, total_tokens)


def train():
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, running_tokens = 0.0, 0

        for src, src_len, decoder_input, target in train_loader:
            src = src.to(DEVICE)
            src_len = src_len.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            target = target.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            dec_in_trim, tgt_trim = _trim_decoder(decoder_input, target)
            logits = model(src, src_len, dec_in_trim, teacher_forcing=TEACHER_FORCING)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_trim.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            n_tokens = (tgt_trim != PAD_IDX).sum().item()
            running_loss += loss.item() * n_tokens
            running_tokens += n_tokens

        train_loss = running_loss / max(1, running_tokens)
        eval_loss = run_eval(eval_loader)
        print(
            f"Epoch {epoch:02d} | train xent/token: {train_loss:.6f} | eval xent/token: {eval_loss:.6f}"
        )


train()

# --------------------------------------------------------
# Inference
# --------------------------------------------------------


def encode_src_str(s: str) -> Tuple[torch.Tensor, torch.Tensor]:
    ids = text_to_idx_list(s, STOI, MAX_LEN_IN)[::-1]  # keep in sync with dataset
    t = torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # [1, T]
    src_len = torch.tensor([t.size(1)], dtype=torch.long)
    return t, src_len


def decode_ids_to_str(ids: torch.Tensor) -> str:
    if ids.dim() == 2:
        ids = ids[0]
    chars = []
    for i in ids.tolist():
        ch = ITOS[i]
        if ch == EOS:
            break
        if ch not in (PAD, SOS):
            chars.append(ch)
    return "".join(chars)


@torch.no_grad()
def predict_one(s: str) -> str:
    model.eval()
    src, src_len = encode_src_str(s)
    src, src_len = src.to(DEVICE), src_len.to(DEVICE)
    pred_ids = model.inference(src, src_len, max_len=MAX_LEN_OUT)
    return decode_ids_to_str(pred_ids.cpu())


@torch.no_grad()
def predict_many(strings: List[str]) -> List[Tuple[str, str]]:
    model.eval()
    outs = []
    for s in strings:
        try:
            pred = predict_one(s)
        except KeyError as e:
            pred = f"<OOV ERROR: {e}>"
        outs.append((s, pred))
    return outs


examples = [
    "10/20/25",
    "20 Oct, 2025",
    "Monday, Oct 20th 2025",
    "2025/10/20",
    "Oct-20-2025",
]

for raw, pred in predict_many(examples):
    print(f"{raw} -> {pred}")
