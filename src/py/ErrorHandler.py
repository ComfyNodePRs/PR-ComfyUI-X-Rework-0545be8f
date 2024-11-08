class ErrorHandler:
    def __init__(self):
        pass
    
    class Errors:
        class ModelError(Exception):
            "Raised when the Node misses a Model"
            pass
        
        class TrainingError(Exception):
            "Raised when a Error happens in the Model Training."
            pass
        
        class NodeError(Exception):
            "Raised when a Error happens in a node."
            pass
        
        class ImageError(Exception):
            "Raised when a Error happens while loading a Image."
            pass