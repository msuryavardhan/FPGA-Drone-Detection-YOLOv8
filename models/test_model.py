import onnxruntime as ort
import numpy as np

def test_onnx_model():
    """Test if the ONNX model can be loaded and run inference"""
    model_path = "best.onnx"
    
    try:
        # Load the model
        print("Loading ONNX model...")
        session = ort.InferenceSession(model_path)
        
        # Get model information
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        
        print(f"Model loaded successfully!")
        print(f"Input name: {input_info.name}")
        print(f"Input shape: {input_info.shape}")
        print(f"Input type: {input_info.type}")
        print(f"Output name: {output_info.name}")
        print(f"Output shape: {output_info.shape}")
        print(f"Output type: {output_info.type}")
        
        # Create dummy input for testing
        input_shape = input_info.shape
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        print(f"\nTesting inference with dummy input of shape: {dummy_input.shape}")
        
        # Run inference
        outputs = session.run(None, {input_info.name: dummy_input})
        
        print(f"Inference successful!")
        print(f"Output shape: {outputs[0].shape}")
        print(f"Output type: {outputs[0].dtype}")
        
        return True
        
    except Exception as e:
        print(f"Error testing model: {e}")
        return False

if __name__ == "__main__":
    test_onnx_model()
