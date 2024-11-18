import torch
import timm
from torch import nn
from typing import Optional, Tuple, Union
class BaseModel(nn.Module):
    """
    A base model class for image classification/regression using pretrained models from the TIMM library.

    Attributes:
    ----------
    model_name : str
        The name of the model architecture to use.
    model : nn.Module
        The internal model instance created from the specified architecture.
    
    Methods:
    -------
    _create_model(model_name: str, dropout_prob: float) -> nn.Module:
        Creates and returns a model with a modified head based on the specified architecture.
    
    load_weights(weight_path: str) -> None:
        Loads weights into the model from the specified file path.
    
    forward(x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        Performs a forward pass through the model, optionally returning feature representations.
    """
    
    def __init__(self, model_name: str, dropout_prob: float = 0.0, weight_path: Optional[str] = None) -> None:
        """
        Initializes the BaseModel.

        Parameters:
        ----------
        model_name : str
            The name of the model architecture to instantiate.
        dropout_prob : float, optional
            The dropout probability to apply to the fully connected layers, by default 0.0.
        weight_path : Optional[str], optional
            Path to the weights file to load into the model, by default None.
        """
        super(BaseModel, self).__init__()
        self.model_name = model_name
        self.model: nn.Module = self._create_model(model_name, dropout_prob)
        if weight_path:
            self.load_weights(weight_path)

    def _create_model(self, model_name: str, dropout_prob: float) -> nn.Module:
        """
        Creates and modifies the model based on the specified architecture.

        Parameters:
        ----------
        model_name : str
            The name of the model architecture to instantiate.
        dropout_prob : float
            The dropout probability to apply to the fully connected layers.

        Returns:
        -------
        nn.Module
            The modified model with the new head.
        
        Raises:
        ------
        ValueError
            If the model architecture is unsupported.
        """
        model: nn.Module = timm.create_model(model_name, pretrained=True)

        # Adjust for different model attribute names
        if 'vgg' in model_name:
            num_ftrs = model.head.fc.in_features
            model.head.fc = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(256, 1)  # Single output for classification/regression
            )
            return model

        # Modify the model's head based on its architecture
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

    def load_weights(self, weight_path: str) -> None:
        """
        Loads weights into the model from the specified file path.

        Parameters:
        ----------
        weight_path : str
            The path to the weights file to load.
        """
        state_dict = torch.load(weight_path, map_location='cpu')
        self.model.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Performs a forward pass through the model.

        Parameters:
        ----------
        x : torch.Tensor
            The input tensor to pass through the model.
        return_features : bool, optional
            If True, returns the feature representations, by default False.

        Returns:
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            The output tensor or a tuple of features and output, based on return_features flag.
        """
        if return_features:
            features = self.model.forward_features(x)
            return features
        else:
            return self.model(x)
