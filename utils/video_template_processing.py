"""
Video Template Processing - Handle video templates (replacing GIF functionality)
"""

import cv2
import os
import numpy as np
from utils.green_screen_detection import create_green_screen_mask
from utils.video_processing import process_frame_with_green_screen

def extract_video_frames(video_path, max_frames=50):
    """Extract frames from video file."""
    try:
        print(f"🎬 Extracting frames from video: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return [], []
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video info: {total_frames} frames at {fps:.1f}fps")
        
        frames = []
        frame_durations = []
        frame_duration = int(1000 / fps) if fps > 0 else 100  # ms per frame
        
        # Limit frames if specified
        if max_frames and total_frames > max_frames:
            frame_step = total_frames // max_frames
            print(f"🔧 Limiting to {max_frames} frames (every {frame_step} frame)")
        else:
            frame_step = max(1, total_frames // 50)  # Always limit to reasonable number
            max_frames = min(total_frames, 50)
            print(f"🔧 Memory optimization: limiting to {max_frames} frames")
        
        frame_count = 0
        extracted_count = 0
        
        while extracted_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_step == 0:
                frames.append(frame)
                frame_durations.append(frame_duration * frame_step)
                extracted_count += 1
                
                if extracted_count % 50 == 0:
                    print(f"📊 Extracted {extracted_count}/{max_frames} frames")
            
            frame_count += 1
        
        cap.release()
        
        print(f"✅ Extracted {len(frames)} frames from video")
        return frames, frame_durations
        
    except Exception as e:
        print(f"❌ Error extracting video frames: {e}")
        return [], []

def process_video_with_video_template_streaming(template_path, video_path, output_path, text_settings):
    """Process video with video template using streaming approach to save RAM."""
    print(f"🎬 Processing video with video template (Streaming Mode)")
    print(f"   Template: {os.path.basename(template_path)}")
    print(f"   Video: {os.path.basename(video_path)}")
    print(f"   Output: {os.path.basename(output_path)}")
    
    # Create temporary output in current working directory
    current_dir = os.getcwd()
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    temp_output = os.path.join(current_dir, f"temp_{base_name}_{os.getpid()}.mp4")
    
    print(f"🔧 Temp file will be created at: {temp_output}")
    
    # Open template video for streaming
    template_cap = cv2.VideoCapture(template_path)
    if not template_cap.isOpened():
        print("❌ Could not open template video")
        return False
    
    # Get template properties
    template_fps = template_cap.get(cv2.CAP_PROP_FPS)
    template_total_frames = int(template_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Open input video
    input_cap = cv2.VideoCapture(video_path)
    if not input_cap.isOpened():
        print("❌ Could not open input video")
        template_cap.release()
        return False
    
    # Get input video properties
    input_fps = int(input_cap.get(cv2.CAP_PROP_FPS))
    input_total_frames = int(input_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Template: {template_total_frames} frames at {template_fps:.1f}fps")
    print(f"📹 Input: {input_total_frames} frames at {input_fps}fps")
    print(f"🔇 Template audio will be muted (using input video audio only)")
    
    # Setup output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(temp_output, fourcc, input_fps, (1080, 1920))
    
    if not out.isOpened():
        print(f"❌ Could not create temp output file: {temp_output}")
        template_cap.release()
        input_cap.release()
        return False
    
    # Process frames with streaming approach
    processed_frames = 0
    template_frame_index = 0
    
    print(f"🔄 Processing {input_total_frames} frames (streaming mode)...")
    
    try:
        while processed_frames < input_total_frames:
            # Read input frame
            ret_input, input_frame = input_cap.read()
            if not ret_input:
                break
            
            # Read template frame (cycle through template)
            if template_frame_index >= template_total_frames:
                template_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                template_frame_index = 0
            
            ret_template, template_frame = template_cap.read()
            if not ret_template:
                # Reset template to beginning
                template_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_template, template_frame = template_cap.read()
                if not ret_template:
                    print("❌ Could not read template frame")
                    break
            
            # Resize template frame
            template_frame = cv2.resize(template_frame, (1080, 1920))
            
            # Create mask and process frame
            from utils.green_screen_detection import create_green_screen_mask
            from utils.video_processing import process_frame_with_green_screen
            
            template_mask = create_green_screen_mask(template_frame)
            processed_frame = process_frame_with_green_screen(
                template_frame, input_frame, template_mask
            )
            
            # Add text overlay if enabled
            if text_settings and text_settings['enabled']:
                from utils.video_processor_core import VideoProcessorCore
                processor = VideoProcessorCore(None)
                video_name = os.path.basename(video_path)
                processed_frame = processor.add_text_overlay(processed_frame, video_name, text_settings)
            
            # Ensure frame is correct size
            if processed_frame.shape[:2] != (1920, 1080):
                processed_frame = cv2.resize(processed_frame, (1080, 1920))
            
            out.write(processed_frame)
            processed_frames += 1
            template_frame_index += 1
            
            if processed_frames % 30 == 0:
                progress = (processed_frames / input_total_frames) * 100
                print(f"📊 Progress: {processed_frames}/{input_total_frames} frames ({progress:.1f}%)")
        
        print(f"✅ Video processing completed: {processed_frames} frames")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        template_cap.release()
        input_cap.release()
        out.release()
    
    # FIXED: Return temp file path instead of moving it immediately
    if processed_frames > 0:
        print(f"✅ Temp video created successfully: {temp_output}")
        return temp_output  # Return temp file path for audio processing
    else:
        # Remove temp file if processing failed
        try:
            if os.path.exists(temp_output):
                os.remove(temp_output)
        except:
            pass
        return False

def process_video_with_video_template(template_path, video_path, output_path, text_settings):
    """Process video with video template - creates MP4 output."""
    # Use streaming approach to save RAM
    return process_video_with_video_template_streaming(template_path, video_path, output_path, text_settings)

def process_image_with_video_template(template_path, image_path, output_path, text_settings, duration=5):
    """Process image with video template - creates MP4 output."""
    print(f"🖼️ Processing image with video template -> MP4 output")
    print(f"   Template: {os.path.basename(template_path)}")
    print(f"   Image: {os.path.basename(image_path)}")
    print(f"   Output: {os.path.basename(output_path)}")
    print(f"   Duration: {duration} seconds")
    
    # Extract template frames
    template_frames, template_durations = extract_video_frames(template_path)
    if not template_frames:
        print("❌ Could not extract template frames")
        return False
    
    print(f"✅ Template loaded: {len(template_frames)} frames")
    
    # Load input image
    input_image = cv2.imread(image_path)
    if input_image is None:
        print("❌ Could not load input image")
        return False
    
    # FIXED: Pre-process input image to prevent shaking/glitch
    # Get template dimensions from first frame
    template_height, template_width = template_frames[0].shape[:2]
    # Resize input image to match template dimensions first for consistent processing
    input_image_processed = cv2.resize(input_image, (template_width, template_height), interpolation=cv2.INTER_AREA)
    
    # Calculate output parameters
    fps = 30  # Standard FPS for image-to-video
    total_output_frames = duration * fps
    template_frame_count = len(template_frames)
    
    print(f"🎯 Output: {total_output_frames} frames at {fps}fps")
    
    # Setup output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (1080, 1920))
    
    if not out.isOpened():
        print(f"❌ Could not create output file: {output_path}")
        return False
    
    # Process frames
    processed_frames = 0
    
    print(f"🔄 Creating {total_output_frames} frames from static image...")
    
    try:
        for frame_index in range(total_output_frames):
            # Get current template frame (cycle through template frames)
            template_frame_index = frame_index % template_frame_count
            current_template_frame = template_frames[template_frame_index]
            
            # Resize template frame to output dimensions
            current_template_frame = cv2.resize(current_template_frame, (1080, 1920))
            
            # Create mask from current template frame
            template_mask = create_green_screen_mask(current_template_frame)
            
            # FIXED: Process frame with green screen using consistently sized image
            processed_frame = process_frame_with_green_screen(
                current_template_frame, input_image_processed, template_mask
            )
            
            # Add text overlay if enabled
            if text_settings and text_settings['enabled']:
                from utils.video_processor_core import VideoProcessorCore
                processor = VideoProcessorCore(None)
                image_name = os.path.basename(image_path)
                processed_frame = processor.add_text_overlay(processed_frame, image_name, text_settings)
            
            # Ensure frame is correct size
            if processed_frame.shape[:2] != (1920, 1080):
                processed_frame = cv2.resize(processed_frame, (1080, 1920))
            
            out.write(processed_frame)
            processed_frames += 1
            
            if processed_frames % 60 == 0:
                progress = (processed_frames / total_output_frames) * 100
                print(f"📊 Progress: {processed_frames}/{total_output_frames} frames ({progress:.1f}%)")
        
        print(f"✅ Image-to-video processing completed: {processed_frames} frames")
        return True
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return False
    
    finally:
        out.release()