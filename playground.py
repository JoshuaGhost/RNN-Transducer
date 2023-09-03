# %%
import datasets
from loss import Loss

ds = datasets.load_dataset("timit_asr", data_dir="./files/timit_asr")
# %%
from train import OPT, Trainer, load_model
from train import get_tokenizer
from train import get_hf_dataloader
from hprams import hprams

tokenizer = get_tokenizer()
phi_idx = tokenizer.special_tokens.phi_id
pad_idx = tokenizer.special_tokens.pad_id
sos_idx = tokenizer.special_tokens.sos_id
vocab_size = tokenizer.vocab_size

train_loader = get_hf_dataloader(ds["train"], tokenizer)
test_loader = get_hf_dataloader(ds["test"], tokenizer)
# %%
criterion = Loss(phi_idx)
model = load_model(vocab_size, pad_idx=pad_idx, phi_idx=phi_idx, sos_idx=sos_idx)
optimizer = OPT[hprams.training.optimizer](
    model.parameters(),
    lr=hprams.training.optim.learning_rate,
    momentum=hprams.training.optim.momentum,
)
trainer = Trainer(
    criterion=criterion,
    optimizer=optimizer,
    model=model,
    device=hprams.device,
    train_loader=train_loader,
    test_loader=test_loader,
    epochs=hprams.training.epochs,
    length_multiplier=hprams.length_multiplier,
)

# %%
trainer.fit()
# %%
