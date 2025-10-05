# model/architecture/m_judge.py

"""
Juez Final - Regresión Logística
Toma las predicciones de los 4 especialistas y hace la decisión final.
"""

# Nota: El juez usa sklearn LogisticRegression, no PyTorch
# Se carga con joblib en lugar de torch.load

class JudgeModel:
    """
    Wrapper para el modelo del juez (Regresión Logística).
    Mantiene consistencia con la interfaz de los otros modelos.
    """
    def __init__(self):
        self.model = None
        self.is_sklearn = True
    
    def load_state_dict(self, model):
        """Carga el modelo de sklearn."""
        self.model = model
    
    def eval(self):
        """Modo evaluación (no hace nada para sklearn)."""
        pass
    
    def predict_proba(self, X):
        """Predice probabilidades."""
        if self.model is None:
            raise ValueError("Modelo no cargado")
        return self.model.predict_proba(X)
    
    def predict(self, X):
        """Predice clases."""
        if self.model is None:
            raise ValueError("Modelo no cargado")
        return self.model.predict(X)
