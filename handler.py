import runpod
import torch
import os
import io
import base64
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
import numpy as np
from typing import Dict, Any, Optional, List
import tempfile
import imageio
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
pipeline = None

def get_device():
    """Get available device with fallback"""
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    else:
        return "cpu"

def load_wan21_pipeline():
    """Load WAN 2.1 Pipeline with fixed device handling"""
    global pipeline
    
    try:
        logger.info("Loading WAN 2.1 (Stable Video Diffusion) pipeline...")
        
        device = get_device()
        logger.info(f"Using device: {device}")
        
        # Load the pipeline
        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32,
            variant="fp16" if device.startswith('cuda') else None,
            use_safetensors=True
        )
        
        pipeline = pipeline.to(device)
        
        # Memory optimizations (only if CUDA)
        if device.startswith('cuda'):
            try:
                pipeline.enable_model_cpu_offload()
                pipeline.enable_vae_slicing()
                logger.info("Memory optimizations enabled")
            except Exception as e:
                logger.warning(f"Could not enable some optimizations: {e}")
        
        logger.info("WAN 2.1 pipeline loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error loading WAN 2.1 pipeline: {str(e)}")
        return False

def create_initial_image(prompt: str, width: int = 1024, height: int = 576) -> Image.Image:
    """Create initial image for video generation"""
    try:
        # Create a simple gradient based on prompt
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Generate gradient based on prompt keywords
        if "sunset" in prompt.lower() or "orange" in prompt.lower():
            # Sunset colors
            for y in range(height):
                ratio = y / height
                img_array[y, :, 0] = int(255 * (1 - ratio * 0.3))  # Red
                img_array[y, :, 1] = int(150 * (1 - ratio * 0.5))  # Green
                img_array[y, :, 2] = int(50 + 100 * ratio)         # Blue
        elif "ocean" in prompt.lower() or "water" in prompt.lower() or "blue" in prompt.lower():
            # Ocean colors
            for y in range(height):
                ratio = y / height
                img_array[y, :, 0] = int(30 + 50 * ratio)          # Red
                img_array[y, :, 1] = int(100 + 100 * (1 - ratio)) # Green
                img_array[y, :, 2] = int(200 + 55 * (1 - ratio))  # Blue
        elif "forest" in prompt.lower() or "green" in prompt.lower():
            # Forest colors
            for y in range(height):
                ratio = y / height
                img_array[y, :, 0] = int(50 + 30 * ratio)          # Red
                img_array[y, :, 1] = int(120 + 100 * (1 - ratio)) # Green
                img_array[y, :, 2] = int(50 + 40 * ratio)          # Blue
        else:
            # Default neutral gradient
            for y in range(height):
                ratio = y / height
                gray_val = int(100 + 100 * (1 - ratio))
                img_array[y, :] = [gray_val, gray_val, gray_val]
        
        return Image.fromarray(img_array)
        
    except Exception as e:
        logger.error(f"Error creating initial image: {str(e)}")
        # Fallback: solid gray image
        img_array = np.full((height, width, 3), 128, dtype=np.uint8)
        return Image.fromarray(img_array)

def generate_video(
    prompt: str,
    image: Optional[Image.Image] = None,
    num_frames: int = 25,
    num_inference_steps: int = 25,
    min_guidance_scale: float = 1.0,
    max_guidance_scale: float = 3.0,
    fps: int = 6,
    motion_bucket_id: int = 127,
    noise_aug_strength: float = 0.02,
    decode_chunk_size: int = 8,
    seed: Optional[int] = None,
    width: int = 1024,
    height: int = 576
) -> List[np.ndarray]:
    """Generate video with WAN 2.1"""
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    try:
        # Create initial image if none provided
        if image is None:
            logger.info("Creating initial image from prompt...")
            image = create_initial_image(prompt, width, height)
        
        # Ensure correct dimensions
        image = image.resize((width, height))
        
        logger.info(f"Generating video with {num_frames} frames...")
        
        # Generate video frames
        frames = pipeline(
            image=image,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            min_guidance_scale=min_guidance_scale,
            max_guidance_scale=max_guidance_scale,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            decode_chunk_size=decode_chunk_size,
        ).frames[0]
        
        # Convert PIL Images to numpy arrays
        frame_arrays = [np.array(frame) for frame in frames]
        
        logger.info(f"Successfully generated {len(frame_arrays)} frames")
        return frame_arrays
        
    except Exception as e:
        logger.error(f"Error generating video: {str(e)}")
        raise e

def save_video_file(frames: List[np.ndarray], output_path: str, fps: int = 6) -> str:
    """Save frames as video file"""
    try:
        logger.info(f"Saving video with {len(frames)} frames at {fps} FPS...")
        
        imageio.mimsave(
            output_path,
            frames,
            fps=fps,
            quality=8,
            macro_block_size=1
        )
        
        logger.info("Video saved successfully!")
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving video: {str(e)}")
        raise e

def encode_video_base64(file_path: str) -> str:
    """Encode video as base64"""
    try:
        with open(file_path, 'rb') as f:
            video_data = f.read()
        
        video_base64 = base64.b64encode(video_data).decode('utf-8')
        return f"data:video/mp4;base64,{video_base64}"
        
    except Exception as e:
        logger.error(f"Error encoding video: {str(e)}")
        raise e

def cleanup_memory():
    """Clean up GPU memory"""
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        logger.warning(f"Memory cleanup warning: {e}")

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Main RunPod handler"""
    
    try:
        job_input = job.get('input', {})
        
        # Extract parameters with defaults
        prompt = job_input.get('prompt', 'A beautiful cinematic scene')
        num_frames = min(job_input.get('num_frames', 16), 25)  # Limit frames
        num_inference_steps = job_input.get('num_inference_steps', 25)
        min_guidance_scale = job_input.get('min_guidance_scale', 1.0)
        max_guidance_scale = job_input.get('max_guidance_scale', 3.0)
        fps = job_input.get('fps', 6)
        motion_bucket_id = job_input.get('motion_bucket_id', 127)
        noise_aug_strength = job_input.get('noise_aug_strength', 0.02)
        decode_chunk_size = job_input.get('decode_chunk_size', 4)  # Reduced for memory
        seed = job_input.get('seed', None)
        width = job_input.get('width', 1024)
        height = job_input.get('height', 576)
        
        # Handle input image if provided
        input_image = None
        if 'image_base64' in job_input and job_input['image_base64']:
            try:
                image_data = base64.b64decode(job_input['image_base64'].split(',')[-1])
                input_image = Image.open(io.BytesIO(image_data))
                logger.info("Using provided input image")
            except Exception as e:
                logger.warning(f"Could not decode input image: {str(e)}")
        
        logger.info(f"Starting video generation: '{prompt}'")
        logger.info(f"Parameters: {num_frames} frames, {width}x{height}, motion={motion_bucket_id}")
        
        # Clean memory before generation
        cleanup_memory()
        
        # Generate video
        frames = generate_video(
            prompt=prompt,
            image=input_image,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            min_guidance_scale=min_guidance_scale,
            max_guidance_scale=max_guidance_scale,
            fps=fps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            decode_chunk_size=decode_chunk_size,
            seed=seed,
            width=width,
            height=height
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            video_path = tmp_file.name
        
        save_video_file(frames, video_path, fps)
        
        # Encode as base64
        video_base64 = encode_video_base64(video_path)
        
        # Cleanup
        os.unlink(video_path)
        cleanup_memory()
        
        return {
            "success": True,
            "video_base64": video_base64,
            "frames_generated": len(frames),
            "parameters": {
                "prompt": prompt,
                "num_frames": num_frames,
                "width": width,
                "height": height,
                "fps": fps,
                "motion_bucket_id": motion_bucket_id
            },
            "model": "Stable Video Diffusion img2vid-xt"
        }
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        cleanup_memory()
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "model": "Stable Video Diffusion img2vid-xt"
        }

if __name__ == "__main__":
    logger.info("Initializing WAN 2.1 handler...")
    
    # Print system info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    if load_wan21_pipeline():
        logger.info("Starting RunPod serverless handler...")
        runpod.serverless.start({"handler": handler})
    else:
        logger.error("Failed to load pipeline. Exiting.")
        exit(1)
