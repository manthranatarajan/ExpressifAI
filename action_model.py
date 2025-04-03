import torch
import torch.nn.functional as F

def load_action_model(device):
    
    # Load the SlowFast model
    model_name = "slowfast_r50"
    model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)
    model = model.to(device)
    model.eval()
    return model

def preprocess_frames(frames):
    tensors = []
    for frame in frames:
        tensor = torch.from_numpy(frame).float() / 255.0  # Normalize to [0, 1]
        
        if len(tensor.shape) == 2:  # Grayscale image (H, W)
            tensor = tensor.unsqueeze(2)  # Add channel dimension -> (H, W, 1)
        
        if tensor.shape[2] == 1:  # Grayscale image (H, W, 1)
            tensor = tensor.repeat(1, 1, 3)  # Repeat channel -> (H, W, 3)

        if tensor.shape[2] != 3:  # Validate number of channels
            raise ValueError(f"Unexpected number of channels: {tensor.shape[2]}")

        # permute to PyTorch format
        tensor = tensor.permute(2, 0, 1)

        # Resize tensor to match model input size (224x224)
        tensor = F.interpolate(tensor.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
        tensors.append(tensor)

    return tensors

def recognize_action(frames, device, action_model):


   # Perform action recognition using the SlowFast model.
    try:
        # Preprocess frames
        processed_frames = preprocess_frames(frames)

        # Ensure enough frames for SlowFast model (minimum 32 for fast_pathway)
        if len(processed_frames) < 32:
            print(f"Insufficient frames ({len(processed_frames)}). Padding to meet minimum requirement.")
            while len(processed_frames) < 32:
                processed_frames.append(processed_frames[-1])  # Duplicate last frame

        # Verify that all frames have consistent dimensions
        for i, frame in enumerate(processed_frames):
            print(f"Frame {i} shape: {frame.shape}")

        # Stack frames into a single tensor
        video_tensor = torch.stack(processed_frames)

        # Prepare fast pathway first
        fast_pathway = video_tensor[:32]  # Use first 32 frames
        fast_pathway = fast_pathway.unsqueeze(0)  # Add batch dimension
        fast_pathway = fast_pathway.permute(0, 2, 1, 3, 4)  # batch, channel, time, height, width

        # Prepare slow pathway
        slow_pathway = video_tensor[:32:4]  # Take 8 frames (32/4) for slow pathway
        slow_pathway = slow_pathway.unsqueeze(0)  # Add batch dimension
        slow_pathway = slow_pathway.permute(0, 2, 1, 3, 4)  # batch, channel, time, height, width

        # Move to device and create input list
        inputs = [slow_pathway.to(device), fast_pathway.to(device)]

        print(f"Input shapes to model: {[i.shape for i in inputs]}")

        # Pass inputs through the action recognition model
        with torch.no_grad():
            preds = action_model(inputs)

        # Apply softmax to get probabilities
        post_act = torch.nn.Softmax(dim=1)
        preds = post_act(preds)

        # Get top predicted actions
        pred_classes = preds.topk(k=5).indices.squeeze(0).tolist()

        # Map predicted classes to readable labels
        kinetics_id_to_classname = {i: f"Action {i}" for i in range(400)}  
        pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]

        return pred_class_names

    except Exception as e:
        print(f"Error during action recognition: {e}")
        return []
