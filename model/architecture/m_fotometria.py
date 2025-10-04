
import torch
import torch.nn as nn

class FotometriaNet(nn.Module):
    """
    Red Neuronal para analizar las características de fotometría de un tránsito.
    Evalúa la calidad, forma y claridad de la señal.
    """
    def __init__(self, input_features):
        """
        Inicializa las capas de la red.
        Args:
            input_features (int): El número de columnas de entrada para esta red.
        """
        super(FotometriaNet, self).__init__()
        # Definimos la arquitectura
        self.network = nn.Sequential(
            nn.Linear(input_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        """
        Define el paso hacia adelante (cómo fluyen los datos).
        """
        return self.network(x)