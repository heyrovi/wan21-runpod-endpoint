import runpod
import torch
import os
import io
import base64
from PIL import Image, ImageEnhance, ImageFilter
from diffusers import StableVideoDiffusionPipeline, DiffusionPipeline
import numpy as np
from typing import Dict, Any, Optional, List
import tempfile
import imageio
import logging
import random

# Setup loggingk
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
video_pipeline = None
image_pipeline = None

def get_device():
    """Get available device with fallback"""
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    else:
        return "cpu"

def load_pipelines():
    """Load both SDXL and WAN 2.1 pipelines"""
    global video_pipeline, image_pipeline
    
    try:
        device = get_device()
        logger.info(f"Using device: {device}")
        
        # Load SDXL for cinematic initial images
        logger.info("Loading SDXL for cinematic image generation...")
        image_pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32,
            use_safetensors=True
        )
        image_pipeline = image_pipeline.to(device)
        
        # Load WAN 2.1 for video generation
        logger.info("Loading WAN 2.1 (Stable Video Diffusion) pipeline...")
        video_pipeline = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32,
            variant="fp16" if device.startswith('cuda') else None,
            use_safetensors=True
        )
        video_pipeline = video_pipeline.to(device)
        
        # Memory optimizations
        if device.startswith('cuda'):
            try:
                image_pipeline.enable_model_cpu_offload()
                video_pipeline.enable_model_cpu_offload()
                video_pipeline.enable_vae_slicing()
                logger.info("Memory optimizations enabled")
            except Exception as e:
                logger.warning(f"Could not enable some optimizations: {e}")
        
        logger.info("All pipelines loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error loading pipelines: {str(e)}")
        return False

def enhance_prompt_for_cinematic_quality(prompt: str) -> str:
    """Transform user prompt into cinematic masterpiece prompt"""
    
    # Cinematic enhancement keywords
    cinematic_styles = [
        "cinematic masterpiece",
        "award-winning cinematography", 
        "hollywood blockbuster style",
        "epic cinematic shot",
        "professional film production"
    ]
    
    lighting_styles = [
        "dramatic lighting",
        "golden hour lighting", 
        "volumetric lighting",
        "cinematic lighting",
        "professional studio lighting"
    ]
    
    quality_enhancers = [
        "8K ultra high definition",
        "hyper realistic",
        "photorealistic",
        "extremely detailed",
        "sharp focus",
        "crystal clear",
        "professional photography"
    ]
    
    camera_movements = [
        "dynamic camera movement",
        "smooth camera pan",
        "cinematic camera work",
        "professional camera operation"
    ]
    
    # Select random enhancements for variety
    style = random.choice(cinematic_styles)
    lighting = random.choice(lighting_styles)
    quality = random.choice(quality_enhancers)
    camera = random.choice(camera_movements)
    
    # Build enhanced prompt
    enhanced_prompt = f"{prompt}, {style}, {lighting}, {quality}, {camera}, film grain, depth of field, bokeh effect, color graded, post-processed"
    
    return enhanced_prompt

def get_optimal_video_settings(prompt: str) -> Dict[str, Any]:
    """Get optimal settings based on prompt content"""
    
    settings = {
        "motion_bucket_id": 127,
        "min_guidance_scale": 1.0,
        "max_guidance_scale": 3.0,
        "noise_aug_strength": 0.02,
        "num_inference_steps": 30
    }
    
    prompt_lower = prompt.lower()
    
    # Action scenes - more motion
    if any(word in prompt_lower for word in ["action", "fast", "running", "flying", "explosion", "chase", "fight"]):
        settings["motion_bucket_id"] = 180
        settings["max_guidance_scale"] = 4.0
        settings["noise_aug_strength"] = 0.05
        
    # Nature/landscape - moderate motion
    elif any(word in prompt_lower for word in ["landscape", "nature", "mountain", "ocean", "forest", "sky"]):
        settings["motion_bucket_id"] = 100
        settings["min_guidance_scale"] = 1.5
        settings["max_guidance_scale"] = 3.5
        
    # Portrait/close-up - minimal motion
    elif any(word in prompt_lower for word in ["portrait", "face", "close-up", "person", "character"]):
        settings["motion_bucket_id"] = 80
        settings["max_guidance_scale"] = 2.5
        settings["noise_aug_strength"] = 0.01
        
    # Dramatic scenes - controlled motion
    elif any(word in prompt_lower for word in ["dramatic", "epic", "cinematic", "movie", "film"]):
        settings["motion_bucket_id"] = 140
        settings["min_guidance_scale"] = 1.2
        settings["max_guidance_scale"] = 3.8
        settings["num_inference_steps"] = 35
        
    # Animals - natural motion
    elif any(word in prompt_lower for word in ["cat", "dog", "bird", "animal", "wildlife"]):
        settings["motion_bucket_id"] = 120
        settings["max_guidance_scale"] = 3.2
    
    return settings

def create_cinematic_initial_image(prompt: str, width: int = 1024, height: int = 576) -> Image.Image:
    """Create Hollywood-quality initial image using SDXL"""
    try:
        logger.info(f"Creating cinematic initial image for: '{prompt}'")
        
        # Enhance prompt for maximum quality
        enhanced_prompt = enhance_prompt_for_cinematic_quality(prompt)
        
        logger.info(f"Enhanced prompt: '{enhanced_prompt}'")
        
        # Negative prompt for maximum quality
        negative_prompt = (
            "blurry, low quality, pixelated, compressed, low-res, poor quality, "
            "distorted, ugly, bad anatomy, deformed, amateur, snapshot, "
            "watermark, signature, text, logo, oversaturated, underexposed"
        )
        
        # Generate with SDXL
        image = image_pipeline(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=40,  # Higher for quality
            guidance_scale=8.5,      # Higher for prompt adherence
        ).images[0]
        
        # Post-process image for even better quality
        image = enhance_image_quality(image)
        
        logger.info("Cinematic initial image created successfully!")
        return image
        
    except Exception as e:
        logger.error(f"Error creating cinematic image: {str(e)}")
        raise e

def enhance_image_quality(image: Image.Image) -> Image.Image:
    """Apply post-processing to enhance image quality"""
    try:
        # Sharpen the image
        sharpening_filter = ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
        image = image.filter(sharpening_filter)
        
        # Enhance contrast slightly
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        
        # Enhance color saturation slightly
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.05)
        
        return image
        
    except Exception as e:
        logger.warning(f"Could not enhance image quality: {e}")
        return image

def generate_hollywood_video(
    prompt: str,
    image: Optional[Image.Image] = None,
    num_frames: int = 25,
    fps: int = 8,
    seed: Optional[int] = None,
    width: int = 1024,
    height: int = 576,
    quality_mode: str = "ultra"  # ultra, high, balanced
) -> List[np.ndarray]:
    """Generate Hollywood-quality video with WAN 2.1"""
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    try:
        # Get optimal settings for this type of content
        settings = get_optimal_video_settings(prompt)
        
        # Adjust settings based on quality mode
        if quality_mode == "ultra":
            settings["num_inference_steps"] = max(35, settings["num_inference_steps"])
            settings["decode_chunk_size"] = 4  # Higher quality, slower
        elif quality_mode == "high":
            settings["num_inference_steps"] = max(30, settings["num_inference_steps"])
            settings["decode_chunk_size"] = 6
        else:  # balanced
            settings["decode_chunk_size"] = 8
        
        # Create cinematic initial image if none provided
        if image is None:
            image = create_cinematic_initial_image(prompt, width, height)
        
        # Ensure correct dimensions
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        
        logger.info(f"Generating Hollywood-quality video with {num_frames} frames...")
        logger.info(f"Settings: motion={settings['motion_bucket_id']}, guidance={settings['min_guidance_scale']}-{settings['max_guidance_scale']}")
        
        # Generate video frames with optimized settings
        frames = video_pipeline(
            image=image,
            num_frames=num_frames,
            num_inference_steps=settings["num_inference_steps"],
            min_guidance_scale=settings["min_guidance_scale"],
            max_guidance_scale=settings["max_guidance_scale"],
            motion_bucket_id=settings["motion_bucket_id"],
            noise_aug_strength=settings["noise_aug_strength"],
            decode_chunk_size=settings["decode_chunk_size"],
        ).frames[0]
        
        # Convert PIL Images to numpy arrays
        frame_arrays = [np.array(frame) for frame in frames]
        
        logger.info(f"Successfully generated {len(frame_arrays)} Hollywood-quality frames")
        return frame_arrays
        
    except Exception as e:
        logger.error(f"Error generating Hollywood video: {str(e)}")
        raise e

def save_video_file(frames: List[np.ndarray], output_path: str, fps: int = 8) -> str:
    """Save frames as high-quality video file"""
    try:
        logger.info(f"Saving Hollywood-quality video with {len(frames)} frames at {fps} FPS...")
        
        # Higher quality settings
        imageio.mimsave(
            output_path,
            frames,
            fps=fps,
            quality=9,           # Maximum quality
            macro_block_size=1,  # Better quality
            ffmpeg_params=['-crf', '18', '-preset', 'slow']  # High quality encoding
        )
        
        logger.info("Hollywood-quality video saved successfully!")
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving video: {str(e)}")
        # Fallback to standard quality
        imageio.mimsave(output_path, frames, fps=fps, quality=8)
        return output_path

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
    """Main RunPod handler for Hollywood-quality videos"""
    
    try:
        job_input = job.get('input', {})
        
        # Extract parameters
        prompt = job_input.get('prompt', 'A cinematic masterpiece scene')
        num_frames = min(job_input.get('num_frames', 20), 25)
        fps = job_input.get('fps', 8)
        seed = job_input.get('seed', None)
        width = job_input.get('width', 1024)
        height = job_input.get('height', 576)
        quality_mode = job_input.get('quality_mode', 'ultra')  # ultra, high, balanced
        
        # Handle input image if provided (optional override)
        input_image = None
        if 'image_base64' in job_input and job_input['image_base64']:
            try:
                image_data = base64.b64decode(job_input['image_base64'].split(',')[-1])
                input_image = Image.open(io.BytesIO(image_data))
                logger.info("Using provided input image instead of generating one")
            except Exception as e:
                logger.warning(f"Could not decode input image, will generate one: {str(e)}")
        
        logger.info(f"Starting Hollywood-quality video generation")
        logger.info(f"Prompt: '{prompt}'")
        logger.info(f"Parameters: {num_frames} frames, {width}x{height}, {fps} FPS, quality: {quality_mode}")
        
        # Clean memory before generation
        cleanup_memory()
        
        # Generate Hollywood-quality video
        frames = generate_hollywood_video(
            prompt=prompt,
            image=input_image,
            num_frames=num_frames,
            fps=fps,
            seed=seed,
            width=width,
            height=height,
            quality_mode=quality_mode
        )
        
        # Save to temporary file with high quality
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
                "original_prompt": prompt,
                "enhanced_prompt": enhance_prompt_for_cinematic_quality(prompt),
                "num_frames": num_frames,
                "width": width,
                "height": height,
                "fps": fps,
                "quality_mode": quality_mode
            },
            "model": "Hollywood-Grade: SDXL + WAN 2.1 Ultra",
            "estimated_duration": f"{len(frames) / fps:.1f} seconds"
        }
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        cleanup_memory()
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "model": "Hollywood-Grade: SDXL + WAN 2.1 Ultra"
        }

if __name__ == "__main__":
    logger.info("Initializing Hollywood-Grade Video Generator...")
    
    # Print system info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    if load_pipelines():
        logger.info("Starting Hollywood-Grade RunPod handler...")
        runpod.serverless.start({"handler": handler})
    else:
        logger.error("Failed to load pipelines. Exiting.")
        exit(1)
