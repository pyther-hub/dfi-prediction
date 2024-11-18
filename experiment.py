import wandb
from torch.utils.data import DataLoader
from model_definitions import BaseModel
from training import train_model
from data_split import get_train_test_dataset
from torch import nn, optim

def RunExperiment(
    set_numb: int,
    model_name: str,
    numb_epochs: int,
    lr: float = 0.0001,
    do_log: bool = True,
    aug_level: int = 1
) -> None:
    model = BaseModel(model_name)
    train_dataset, val_dataset = get_train_test_dataset(set_numb, aug_level)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if do_log:
        wandb.init(project="final_project", config={'model': model_name, 'epochs': numb_epochs})
    train_model(model, train_dataloader, val_dataloader, criterion, optimizer, f'{model_name}_set{set_numb}', numb_epochs)
    wandb.finish()

model_names = [
            #    "seresnext50_32x4d", 
            #    "resnet50", 
            #    "resnet18",
            #    "resnet101",
            #    "inception_v3", 
            #    "densenet121", 
               "efficientnet_b0", 
               "efficientnet_b1",
            #    "efficientnet_b2",
            #    "efficientnet_b3",
            #    "vit_base_patch16_224",
            #    "vit_small_patch16_224",
              ]
for model_name in model_names:
    for set_numb in range(1,7):
        RunExperiment(model_name= model_name, set_numb=set_numb, numb_epochs=100, lr=5e-4, do_log=True, aug_level=2)
