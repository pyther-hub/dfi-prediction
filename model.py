import torch
import timm
from torch import nn

class BaseModel(nn.Module):
    def __init__(self, model_name, dropout_prob=0.5, weight_path=None, device=None):
        super(BaseModel, self).__init__()
        self.model_name = model_name
        self.model = self._create_model(model_name, dropout_prob)
        if weight_path:
            self.load_weights(weight_path)
        self.device = device if device else torch.device('cpu')
        self.to(self.device)

    def _create_model(self, model_name, dropout_prob):
        model = timm.create_model(model_name, pretrained=False)
        # Adjust for different model attribute names
        if hasattr(model, 'fc'):
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(256, 1)
            )
        elif hasattr(model, 'classif'):
            num_ftrs = model.classif.in_features
            model.classif = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(256, 1)
            )
        elif hasattr(model, 'classifier'):
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(256, 1)
            )
        elif hasattr(model, 'head'):
            num_ftrs = model.head.in_features
            model.head = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(256, 1)
            )
        else:
            raise ValueError(f"Unsupported model architecture: {model_name}")
        
        return model
    
    def load_weights(self, weight_path):
        state_dict = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def forward(self, x, return_features=False):
        x = x.to(self.device)
        if return_features:
            features = self.model.forward_features(x)
            return features
        else:
            return self.model(x)

# Example usage
model_names = ["seresnext50_32x4d", "resnet50", "inception_v3", "densenet121", "efficientnet_b0", "vit_base_patch16_224", "vit_small_patch16_224"]

