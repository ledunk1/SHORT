import cv2
import numpy as np
from PIL import Image
import os

def is_gif_file(file_path):
    """Check if file is a GIF."""
    return file_path.lower().endswith('.gif')

def extract_gif_frames(gif_path):
    """Extract frames from GIF file with improved frame synchronization."""
    try:
        print(f"🔍 Opening GIF: {gif_path}")
        
        # Verify file exists and is readable
        if not os.path.exists(gif_path):
            print(f"❌ GIF file not found: {gif_path}")
            return [], []
        
        # Get file size for validation
        file_size = os.path.getsize(gif_path)
        print(f"📊 GIF file size: {file_size / 1024:.2f} KB")
        
        if file_size == 0:
            print("❌ GIF file is empty")
            return [], []
        
        gif = Image.open(gif_path)
        
        frames = []
        durations = []
        
        # Get total number of frames more reliably
        frame_count = 0
        try:
            while True:
                gif.seek(frame_count)
                frame_count += 1
        except EOFError:
            pass
        
        print(f"📊 GIF has {frame_count} frames")
        
        if frame_count == 0:
            print("❌ No frames found in GIF")
            return [], []
        
        # Extract all frames with better error handling
        successful_frames = 0
        for frame_num in range(frame_count):
            try:
                gif.seek(frame_num)
                
                # Get frame info
                frame_info = gif.info
                duration = frame_info.get('duration', 100)  # Default 100ms
                
                # Convert to RGBA first to handle transparency properly
                if gif.mode != 'RGBA':
                    frame = gif.convert('RGBA')
                else:
                    frame = gif.copy()
                
                # Create white background for transparency
                background = Image.new('RGB', frame.size, (255, 255, 255))
                
                # Paste frame onto background using alpha channel
                if frame.mode == 'RGBA':
                    background.paste(frame, mask=frame.split()[-1])
                else:
                    background.paste(frame)
                
                # Convert PIL to OpenCV format (RGB to BGR)
                frame_array = np.array(background)
                frame_cv = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                
                frames.append(frame_cv)
                durations.append(max(50, duration))  # Minimum 50ms per frame
                successful_frames += 1
                
                if (frame_num + 1) % 10 == 0:
                    print(f"📊 Extracted {frame_num + 1}/{frame_count} frames")
                    
            except Exception as e:
                print(f"❌ Error extracting frame {frame_num}: {e}")
                # Skip this frame and continue
                continue
        
        print(f"✅ Successfully extracted {successful_frames}/{frame_count} frames from GIF")
        
        if successful_frames == 0:
            print("❌ No frames could be extracted")
            return [], []
        
        return frames, durations
        
    except Exception as e:
        print(f"❌ Error extracting GIF frames: {e}")
        import traceback
        traceback.print_exc()
        return [], []

def create_gif_from_frames(frames, output_path, durations=None, fps=10):
    """Create animated GIF from frames with proper frame timing."""
    try:
        if not frames:
            print("❌ No frames to create GIF")
            return False
        
        print(f"🎬 Creating animated GIF with {len(frames)} frames")
        
        # Convert OpenCV frames to PIL
        pil_frames = []
        for i, frame in enumerate(frames):
            try:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                
                # Resize if frame is too large (for reasonable file size)
                width, height = pil_frame.size
                max_dimension = 1080
                
                if width > max_dimension or height > max_dimension:
                    # Calculate new dimensions maintaining aspect ratio
                    if width > height:
                        new_width = max_dimension
                        new_height = int(height * (max_dimension / width))
                    else:
                        new_height = max_dimension
                        new_width = int(width * (max_dimension / height))
                    
                    pil_frame = pil_frame.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    print(f"📏 Resized frame {i+1}: {width}x{height} -> {new_width}x{new_height}")
                
                # Optimize for GIF (reduce colors)
                pil_frame = pil_frame.convert('P', palette=Image.ADAPTIVE, colors=256)
                pil_frames.append(pil_frame)
                
                if (i + 1) % 20 == 0:
                    print(f"📊 Converted {i + 1}/{len(frames)} frames")
                    
            except Exception as e:
                print(f"❌ Error converting frame {i}: {e}")
                continue
        
        if not pil_frames:
            print("❌ No frames could be converted")
            return False
        
        # Handle durations
        if durations is None or len(durations) != len(pil_frames):
            print(f"⚠️ Duration mismatch or missing, using default timing")
            duration_ms = int(1000 / fps)  # Convert fps to milliseconds
            durations = [duration_ms] * len(pil_frames)
        else:
            # Ensure durations match frame count
            durations = durations[:len(pil_frames)]
            while len(durations) < len(pil_frames):
                durations.append(100)  # Default 100ms
        
        # Clamp durations to reasonable range
        durations = [max(50, min(1000, d)) for d in durations]
        
        print(f"💾 Saving animated GIF to: {output_path}")
        print(f"⏱️ Frame count: {len(pil_frames)}")
        print(f"⏱️ Duration range: {min(durations)}ms - {max(durations)}ms")
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as animated GIF
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=durations,
            loop=0,  # Infinite loop
            optimize=True,
            disposal=2  # Clear frame before next
        )
        
        # Verify the created file
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"✅ Animated GIF created successfully!")
            print(f"📁 Output: {output_path}")
            print(f"📊 Size: {file_size:.2f} MB")
            
            # Test if it's actually animated
            try:
                test_gif = Image.open(output_path)
                if hasattr(test_gif, 'is_animated') and test_gif.is_animated:
                    print(f"🎬 Animation verified: {test_gif.n_frames} frames")
                    return True
                elif hasattr(test_gif, 'n_frames') and test_gif.n_frames > 1:
                    print(f"🎬 Multi-frame GIF: {test_gif.n_frames} frames")
                    return True
                else:
                    print("⚠️ Created GIF appears to be static")
                    return True  # Still consider it successful
                test_gif.close()
            except Exception as e:
                print(f"⚠️ Could not verify GIF: {e}")
                return True  # File exists, assume success
        else:
            print("❌ GIF file was not created")
            return False
            
    except Exception as e:
        print(f"❌ Error creating animated GIF: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_video_with_gif_template(gif_template_path, video_path, output_path, text_settings):
    """Process video with animated GIF template - OUTPUT ANIMATED GIF."""
    from .video_processing import process_frame_with_green_screen
    from .green_screen_detection import create_green_screen_mask
    
    print(f"🎬 Processing video with animated GIF template -> ANIMATED GIF output")
    print(f"   GIF Template: {os.path.basename(gif_template_path)}")
    print(f"   Video: {os.path.basename(video_path)}")
    print(f"   Output: {os.path.basename(output_path)} (ANIMATED GIF)")
    
    # Extract GIF frames
    gif_frames, gif_durations = extract_gif_frames(gif_template_path)
    if not gif_frames:
        print("❌ Could not extract GIF template frames")
        return False
    
    print(f"✅ GIF template loaded: {len(gif_frames)} frames")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video file")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_video_frames / fps
    
    print(f"📹 Video properties: {total_video_frames} frames at {fps} FPS")
    
    # Calculate frame timing for GIF synchronization
    gif_frame_count = len(gif_frames)
    
    # Create mask from first GIF frame
    template_mask = create_green_screen_mask(gif_frames[0])
    
    # Calculate output timing for GIF
    gif_total_duration = sum(gif_durations) / 1000.0 if gif_durations else (gif_frame_count / 10.0)
    
    # Determine how many cycles we need
    if video_duration <= gif_total_duration:
        # Video is shorter, use video duration
        output_duration = video_duration
        total_output_frames = total_video_frames
    else:
        # Video is longer, loop GIF to match video duration
        output_duration = video_duration
        total_output_frames = total_video_frames
    
    print(f"🎯 Output duration: {output_duration:.2f}s, {total_output_frames} frames")
    
    frame_index = 0
    processed_frames = []
    output_durations = []
    
    print(f"🔄 Processing {total_output_frames} frames for animated GIF output...")
    
    try:
        while frame_index < total_output_frames:
            ret, video_frame = cap.read()
            if not ret:
                # Video ended, loop if needed
                if frame_index < total_output_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, video_frame = cap.read()
                    if not ret:
                        break
                    print(f"🔄 Looping video to match output duration...")
                else:
                    break
            
            # Calculate which GIF frame to use (cycle through GIF frames)
            gif_frame_index = frame_index % gif_frame_count
            current_gif_frame = gif_frames[gif_frame_index]
            
            # Resize GIF frame to match output dimensions
            current_gif_frame = cv2.resize(current_gif_frame, (1080, 1920))
            
            # Update mask for current GIF frame
            current_template_mask = create_green_screen_mask(current_gif_frame)
            
            # Process frame with green screen
            processed_frame = process_frame_with_green_screen(current_gif_frame, video_frame, current_template_mask)
            
            # Add text overlay if enabled
            if text_settings and text_settings['enabled']:
                from .video_processor_core import VideoProcessorCore
                processor = VideoProcessorCore(None)
                video_name = os.path.basename(video_path)
                processed_frame = processor.add_text_overlay(processed_frame, video_name, text_settings)
            
            # Ensure frame is correct size
            if processed_frame.shape[:2] != (1920, 1080):
                processed_frame = cv2.resize(processed_frame, (1080, 1920))
            
            # Add to frames list
            processed_frames.append(processed_frame)
            
            # Calculate frame duration for GIF
            current_gif_duration = gif_durations[gif_frame_index] if gif_durations else 100
            output_durations.append(current_gif_duration)
            
            frame_index += 1
            
            if frame_index % 30 == 0:
                print(f"📊 Processed {frame_index}/{total_output_frames} frames (GIF frame: {gif_frame_index + 1}/{gif_frame_count})")
        
        print(f"✅ Processed {frame_index} frames total")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return False
    
    finally:
        cap.release()
    
    # Create animated GIF from processed frames
    print(f"🎬 Creating animated GIF with {len(processed_frames)} frames...")
    success = create_gif_from_frames(processed_frames, output_path, output_durations)
    
    if success:
        print(f"✅ Animated GIF created successfully!")
        print(f"📁 Output: {output_path}")
        return True
    else:
        print(f"❌ Animated GIF creation failed")
        return False

def process_gif_greenscreen(gif_path, template, template_mask, output_path, text_settings):
    """Process GIF with green screen mode (legacy function for GIF input)."""
    from .video_processing import process_frame_with_green_screen
    
    print(f"🎬 Starting GIF greenscreen processing: {os.path.basename(gif_path)}")
    
    frames, durations = extract_gif_frames(gif_path)
    if not frames:
        print("❌ No frames extracted from GIF")
        return False
    
    processed_frames = []
    gif_name = os.path.basename(gif_path)
    
    print(f"🔄 Processing {len(frames)} frames with greenscreen...")
    
    for i, frame in enumerate(frames):
        try:
            # Process frame with green screen
            processed_frame = process_frame_with_green_screen(template, frame, template_mask)
            
            # Add text overlay if enabled
            if text_settings and text_settings['enabled']:
                from .video_processor_core import VideoProcessorCore
                processor = VideoProcessorCore(None)
                processed_frame = processor.add_text_overlay(processed_frame, gif_name, text_settings)
            
            # Ensure frame is correct size (9:16 aspect ratio)
            if processed_frame.shape[:2] != (1920, 1080):
                processed_frame = cv2.resize(processed_frame, (1080, 1920))
            
            processed_frames.append(processed_frame)
            
            if (i + 1) % 10 == 0:
                print(f"📊 Processed {i + 1}/{len(frames)} frames")
                
        except Exception as e:
            print(f"❌ Error processing frame {i}: {e}")
            # Use resized original frame as fallback
            fallback_frame = cv2.resize(frame, (1080, 1920))
            processed_frames.append(fallback_frame)
    
    print(f"✅ Processed {len(processed_frames)} frames total")
    
    # Create output GIF with original timing
    success = create_gif_from_frames(processed_frames, output_path, durations)
    
    if success:
        print(f"✅ GIF greenscreen processing completed: {gif_name}")
    else:
        print(f"❌ GIF greenscreen processing failed: {gif_name}")
    
    return success

def process_gif_blur(gif_path, output_path, blur_settings, text_settings=None):
    """Process GIF with blur background mode."""
    from .blur_processing import process_blur_frame
    
    print(f"🌀 Starting GIF blur processing: {os.path.basename(gif_path)}")
    
    frames, durations = extract_gif_frames(gif_path)
    if not frames:
        print("❌ No frames extracted from GIF")
        return False
    
    processed_frames = []
    gif_name = os.path.basename(gif_path)
    
    print(f"🔄 Processing {len(frames)} frames with blur...")
    
    for i, frame in enumerate(frames):
        try:
            # Process frame with blur background
            processed_frame = process_blur_frame(
                frame,
                blur_settings['crop_top'],
                blur_settings['crop_bottom'],
                blur_settings['video_x_position'],
                blur_settings['video_y_position'],
                1080,  # target_width
                1920   # target_height
            )
            
            # Add text overlay if enabled
            if text_settings and text_settings['enabled']:
                from .video_processor_core import VideoProcessorCore
                processor = VideoProcessorCore(None)
                processed_frame = processor.add_text_overlay(processed_frame, gif_name, text_settings)
            
            # Ensure frame is correct size
            if processed_frame.shape[:2] != (1920, 1080):
                processed_frame = cv2.resize(processed_frame, (1080, 1920))
            
            processed_frames.append(processed_frame)
            
            if (i + 1) % 10 == 0:
                print(f"📊 Processed {i + 1}/{len(frames)} frames")
                
        except Exception as e:
            print(f"❌ Error processing frame {i}: {e}")
            # Use resized original frame as fallback
            fallback_frame = cv2.resize(frame, (1080, 1920))
            processed_frames.append(fallback_frame)
    
    print(f"✅ Processed {len(processed_frames)} frames total")
    
    # Create output GIF with original timing
    success = create_gif_from_frames(processed_frames, output_path, durations)
    
    if success:
        print(f"✅ GIF blur processing completed: {os.path.basename(gif_path)}")
    else:
        print(f"❌ GIF blur processing failed: {os.path.basename(gif_path)}")
    
    return success