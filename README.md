# WAN 2.1 Video Generation Endpoint

RunPod Serverless Endpoint für WAN 2.1 (Stable Video Diffusion) Video-Generierung.

## Features
- High-quality video generation mit WAN 2.1
- Customizable motion control
- Base64 video output
- Memory optimiert für GPU efficiency

## Parameters
- `prompt`: Text description
- `num_frames`: 14-25 frames
- `motion_bucket_id`: 1-255 (motion intensity)
- `fps`: Output frame rate
- `width/height`: Video dimensions (default 1024x576)

## Usage
Send POST request to RunPod endpoint with parameters in `input` field.
