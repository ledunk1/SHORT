"""
Narasi Processing Module
Handles narasi mode processing with video concatenation and template application
"""

import os
import cv2
import numpy as np
import tempfile
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, concatenate_audioclips, CompositeAudioClip
from utils.file_operations import get_video_files, get_audio_files, create_output_folder
from utils.green_screen_detection import create_green_screen_mask
from utils.video_processing import process_frame_with_green_screen

def is_video_file(file_path):
    """Check if file is a video."""
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')
    return file_path.lower().endswith(video_extensions)

def process_concatenated_video_with_video_template(concatenated_video_path, template_path, output_path, text_settings, original_media_name=None):
    """Process concatenated video with video template using streaming approach."""
    print(f"🎬 Processing with video template (streaming mode)")
    print(f"   Template: {os.path.basename(template_path)}")
    print(f"   Video: {os.path.basename(concatenated_video_path)}")
    print(f"   🎵 Preserving original audio from concatenated video")
    
    # Create temporary output
    current_dir = os.getcwd()
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    temp_video_only = os.path.join(current_dir, f"temp_video_only_{base_name}_{os.getpid()}.mp4")
    temp_output = os.path.join(current_dir, f"temp_{base_name}_{os.getpid()}.mp4")
    
    # Open template video
    template_cap = cv2.VideoCapture(template_path)
    if not template_cap.isOpened():
        print("❌ Could not open template video")
        return False
    
    # Get template properties
    template_fps = template_cap.get(cv2.CAP_PROP_FPS)
    template_total_frames = int(template_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Open input video
    input_cap = cv2.VideoCapture(concatenated_video_path)
    if not input_cap.isOpened():
        print("❌ Could not open input video")
        template_cap.release()
        return False
    
    # Get input video properties
    input_fps = int(input_cap.get(cv2.CAP_PROP_FPS))
    input_total_frames = int(input_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Template: {template_total_frames} frames at {template_fps:.1f}fps")
    print(f"📹 Input: {input_total_frames} frames at {input_fps}fps")
    
    # Setup output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_only, fourcc, input_fps, (1080, 1920))
    
    if not out.isOpened():
        print(f"❌ Could not create temp video file: {temp_video_only}")
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
            template_mask = create_green_screen_mask(template_frame)
            processed_frame = process_frame_with_green_screen(
                template_frame, input_frame, template_mask
            )
            
            # Add text overlay if enabled
            if text_settings and text_settings['enabled']:
                from utils.video_processor_core import VideoProcessorCore
                processor = VideoProcessorCore(None)
                # FIXED: Use original media name instead of temp file name
                video_name = original_media_name if original_media_name else os.path.basename(concatenated_video_path)
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
    
    if processed_frames > 0:
        print(f"🎵 Adding original audio back to processed video...")
        
        # Now add the original audio back using MoviePy
        try:
            from moviepy.editor import VideoFileClip, AudioFileClip
            
            # Load processed video (no audio) and original audio
            processed_video = VideoFileClip(temp_video_only)
            original_audio = AudioFileClip(concatenated_video_path)
            
            print(f"📹 Processed video: {processed_video.duration:.2f}s")
            print(f"🎵 Original audio: {original_audio.duration:.2f}s")
            
            # Combine video with original audio
            final_video = processed_video.set_audio(original_audio)
            
            # Write final video with audio
            final_video.write_videofile(temp_output, codec='libx264', audio_codec='aac', logger=None)
            
            # Cleanup
            processed_video.close()
            original_audio.close()
            final_video.close()
            
            # Remove temp video-only file
            if os.path.exists(temp_video_only):
                os.remove(temp_video_only)
            
            print(f"✅ Video with original audio created: {temp_output}")
            return temp_output
            
        except Exception as audio_error:
            print(f"❌ Error adding original audio: {audio_error}")
            # Return video-only file as fallback
            return temp_video_only
    else:
        # Remove temp file if processing failed
        try:
            if os.path.exists(temp_video_only):
                os.remove(temp_video_only)
        except:
            pass
        return False

def process_concatenated_video_with_static_template(concatenated_video_path, template_path, output_path, text_settings, original_media_name=None):
    """Process concatenated video with static template."""
    print(f"🖼️ Processing with static template")
    print(f"   Template: {os.path.basename(template_path)}")
    print(f"   Video: {os.path.basename(concatenated_video_path)}")
    print(f"   🎵 Preserving original audio from concatenated video")
    
    # Load static template
    template = cv2.imread(template_path)
    if template is None:
        print("❌ Could not load static template")
        return False
    
    template = cv2.resize(template, (1080, 1920))
    template_mask = create_green_screen_mask(template)
    
    if np.sum(template_mask) == 0:
        print("❌ No green screen detected in template")
        return False
    
    print(f"✅ Static template loaded with green screen mask")
    
    # Create temporary output
    current_dir = os.getcwd()
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    temp_video_only = os.path.join(current_dir, f"temp_video_only_{base_name}_{os.getpid()}.mp4")
    temp_output = os.path.join(current_dir, f"temp_{base_name}_{os.getpid()}.mp4")
    
    # Open input video
    cap = cv2.VideoCapture(concatenated_video_path)
    if not cap.isOpened():
        print("❌ Could not open concatenated video")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup output writer (video only first)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_only, fourcc, fps, (1080, 1920))
    
    if not out.isOpened():
        print(f"❌ Could not create temp video file: {temp_video_only}")
        cap.release()
        return False
    
    frame_count = 0
    # FIXED: Use original media name instead of temp file name
    video_name = original_media_name if original_media_name else os.path.basename(concatenated_video_path)
    
    print(f"🔄 Processing {total_frames} frames with static template...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with green screen
            processed_frame = process_frame_with_green_screen(template, frame, template_mask)
            
            # Add text overlay if enabled
            if text_settings and text_settings['enabled']:
                from utils.video_processor_core import VideoProcessorCore
                processor = VideoProcessorCore(None)
                processed_frame = processor.add_text_overlay(processed_frame, video_name, text_settings)
            
            # Ensure frame is correct size
            if processed_frame.shape[:2] != (1920, 1080):
                processed_frame = cv2.resize(processed_frame, (1080, 1920))
            
            out.write(processed_frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"📊 Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)")
        
        print(f"✅ Video processing completed: {frame_count} frames")
        
    except Exception as e:
        print(f"❌ Error processing with static template: {e}")
        return False
    
    finally:
        cap.release()
        out.release()
    
    if frame_count > 0:
        print(f"🎵 Adding original audio back to processed video...")
        
        # Now add the original audio back using MoviePy
        try:
            from moviepy.editor import VideoFileClip, AudioFileClip
            
            # Load processed video (no audio) and original audio
            processed_video = VideoFileClip(temp_video_only)
            original_audio = AudioFileClip(concatenated_video_path)
            
            print(f"📹 Processed video: {processed_video.duration:.2f}s")
            print(f"🎵 Original audio: {original_audio.duration:.2f}s")
            
            # Combine video with original audio
            final_video = processed_video.set_audio(original_audio)
            
            # Write final video with audio
            final_video.write_videofile(temp_output, codec='libx264', audio_codec='aac', logger=None)
            
            # Cleanup
            processed_video.close()
            original_audio.close()
            final_video.close()
            
            # Remove temp video-only file
            if os.path.exists(temp_video_only):
                os.remove(temp_video_only)
            
            print(f"✅ Video with original audio created: {temp_output}")
            return temp_output
            
        except Exception as audio_error:
            print(f"❌ Error adding original audio: {audio_error}")
            # Return video-only file as fallback
            return temp_video_only
    else:
        # Remove temp file if processing failed
        try:
            if os.path.exists(temp_video_only):
                os.remove(temp_video_only)
        except:
            pass
        return False

def process_concatenated_video_with_template(concatenated_video_path, template_path, output_path, text_settings, original_media_name=None):
    """Process concatenated video with template (auto-detect video or static)."""
    print(f"📝 Step 2: Processing with template...")
    
    # Check if template is video or static image
    if is_video_file(template_path):
        print(f"🎬 Detected video template: {os.path.basename(template_path)}")
        return process_concatenated_video_with_video_template(
            concatenated_video_path, template_path, output_path, text_settings, original_media_name
        )
    else:
        print(f"🖼️ Detected static template: {os.path.basename(template_path)}")
        return process_concatenated_video_with_static_template(
            concatenated_video_path, template_path, output_path, text_settings, original_media_name
        )

def concatenate_videos_with_moviepy(video_paths, output_path):
    """Concatenate videos using MoviePy with improved error handling."""
    print(f"🔗 Concatenating {len(video_paths)} videos with MoviePy...")
    
    clips = []
    total_duration = 0
    
    try:
        # Load all video clips
        for i, video_path in enumerate(video_paths):
            print(f"📹 Loading video {i+1}/{len(video_paths)}: {os.path.basename(video_path)}")
            
            try:
                clip = VideoFileClip(video_path)
                
                # Get clip info
                duration = clip.duration
                fps = clip.fps
                size = clip.size
                
                print(f"   ✅ Loaded: {duration:.2f}s, {fps}fps, {size}")
                
                clips.append(clip)
                total_duration += duration
                
            except Exception as e:
                print(f"   ❌ Error loading {video_path}: {e}")
                continue
        
        if not clips:
            print("❌ No clips could be loaded")
            return False
        
        print(f"🔗 Concatenating {len(clips)} clips...")
        
        # Concatenate clips
        final_clip = concatenate_videoclips(clips, method="compose")
        
        print(f"💾 Writing concatenated video...")
        
        # Write output
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            logger=None
        )
        
        print(f"✅ Video concatenation completed: {total_duration:.2f} seconds")
        
        # Get final duration
        final_duration = final_clip.duration
        print(f"📹 Concatenated video duration: {final_duration:.2f} seconds")
        
        return final_duration
        
    except Exception as e:
        print(f"❌ Error during concatenation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up clips
        for clip in clips:
            try:
                clip.close()
            except:
                pass
        
        try:
            if 'final_clip' in locals():
                final_clip.close()
        except:
            pass

def add_narasi_audio_to_video(temp_video_path, audio_path, output_path, audio_mode="narasi_only", 
                             narasi_volume=100, original_volume=30):
    """Add narasi audio to processed video with different modes."""
    print(f"🎵 Adding narasi audio...")
    print(f"   Mode: {audio_mode}")
    print(f"   Audio: {os.path.basename(audio_path)}")
    print(f"   Output: {os.path.basename(output_path)}")
    
    video_clip = None
    narasi_audio = None
    original_audio = None
    final_clip = None
    
    try:
        # Load video and narasi audio
        video_clip = VideoFileClip(temp_video_path)
        narasi_audio = AudioFileClip(audio_path)
        
        print(f"📹 Video duration: {video_clip.duration:.2f}s")
        print(f"🎵 Narasi duration: {narasi_audio.duration:.2f}s")
        
        if audio_mode == "narasi_only":
            # Only narasi audio
            narasi_volume_factor = narasi_volume / 100.0
            if narasi_volume_factor != 1.0:
                narasi_audio = narasi_audio.volumex(narasi_volume_factor)
            
            # FIXED: Don't loop narasi audio, just trim if longer than video
            if narasi_audio.duration > video_clip.duration:
                narasi_audio = narasi_audio.subclip(0, video_clip.duration)
                print(f"✂️ Narasi audio trimmed to match video duration: {video_clip.duration:.2f}s")
            else:
                print(f"✅ Narasi audio duration: {narasi_audio.duration:.2f}s (no looping)")
            
            final_clip = video_clip.set_audio(narasi_audio)
            
        elif audio_mode == "mixed_audio":
            # FIXED: Get original audio from the temp video (which contains concatenated video audio)
            original_audio = video_clip.audio
            
            if original_audio is None:
                print("⚠️ No original audio found in processed video, using narasi only")
                narasi_volume_factor = narasi_volume / 100.0
                if narasi_volume_factor != 1.0:
                    narasi_audio = narasi_audio.volumex(narasi_volume_factor)
                
                # FIXED: Don't loop narasi audio
                if narasi_audio.duration > video_clip.duration:
                    narasi_audio = narasi_audio.subclip(0, video_clip.duration)
                
                final_clip = video_clip.set_audio(narasi_audio)
            else:
                print(f"✅ Original audio found: {original_audio.duration:.2f}s")
                
                # Apply volume adjustments
                narasi_volume_factor = narasi_volume / 100.0
                original_volume_factor = original_volume / 100.0
                
                if narasi_volume_factor != 1.0:
                    narasi_audio = narasi_audio.volumex(narasi_volume_factor)
                    print(f"🔊 Narasi volume adjusted to {narasi_volume}%")
                
                if original_volume_factor != 1.0:
                    original_audio = original_audio.volumex(original_volume_factor)
                    print(f"🔊 Original volume adjusted to {original_volume}%")
                
                # FIXED: Don't loop narasi audio, just overlay as is
                if narasi_audio.duration > video_clip.duration:
                    narasi_audio = narasi_audio.subclip(0, video_clip.duration)
                    print(f"✂️ Narasi audio trimmed to match video duration")
                else:
                    print(f"✅ Narasi audio will overlay for {narasi_audio.duration:.2f}s of {video_clip.duration:.2f}s video")
                
                # FIXED: Create composite audio with proper duration handling
                # If narasi is shorter than video, it will naturally end and only original audio continues
                composite_audio = CompositeAudioClip([original_audio, narasi_audio])
                final_clip = video_clip.set_audio(composite_audio)
        
        else:
            print(f"❌ Unknown audio mode: {audio_mode}")
            return False
        
        # Write final video
        print(f"💾 Writing final video with narasi audio...")
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            logger=None
        )
        
        print(f"✅ Narasi audio processing completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error adding narasi audio: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup resources
        try:
            if video_clip:
                video_clip.close()
            if narasi_audio:
                narasi_audio.close()
            if original_audio:
                original_audio.close()
            if final_clip:
                final_clip.close()
        except Exception as cleanup_error:
            print(f"⚠️ Cleanup error: {cleanup_error}")
        
        # Remove temp file
        try:
            if os.path.exists(temp_video_path) and "temp" in os.path.basename(temp_video_path).lower():
                os.remove(temp_video_path)
                print(f"🧹 Temporary file removed: {os.path.basename(temp_video_path)}")
        except Exception as temp_cleanup_error:
            print(f"⚠️ Could not remove temp file: {temp_cleanup_error}")

def process_single_narasi_match(video_files, audio_file, video_folder, audio_folder, 
                               template_path, output_folder, text_settings, gpu_settings,
                               audio_mode="narasi_only", narasi_volume=100, original_volume=30):
    """Process a single narasi match with improved logic."""
    print(f"🎬 Processing single narasi match (Improved Logic)...")
    print(f"   Videos: {len(video_files)} files")
    print(f"   Audio: {audio_file}")
    print(f"   Audio mode: {audio_mode}")
    
    # Get audio duration
    audio_path = os.path.join(audio_folder, audio_file)
    try:
        audio_clip = AudioFileClip(audio_path)
        audio_duration = audio_clip.duration
        audio_clip.close()
        print(f"🎵 Audio duration: {audio_duration:.2f} seconds")
    except Exception as e:
        print(f"❌ Could not get audio duration: {e}")
        return False
    
    # Create temporary concatenated video
    temp_dir = tempfile.gettempdir()
    temp_concat_video = os.path.join(temp_dir, f"temp_concat_{os.getpid()}.mp4")
    
    try:
        print(f"📝 Step 1: Concatenating videos...")
        
        # Get full paths for videos
        video_paths = [os.path.join(video_folder, vf) for vf in video_files]
        
        # Concatenate videos
        video_duration = concatenate_videos_with_moviepy(video_paths, temp_concat_video)
        
        if not video_duration:
            print("❌ Video concatenation failed")
            return False
        
        print(f"🎯 Final duration: {video_duration:.2f}s (video duration - audio as overlay)")
        
        # Create output path
        audio_base_name = os.path.splitext(audio_file)[0]
        output_filename = f"narasi_{audio_base_name}.mp4"
        final_output_path = os.path.join(output_folder, output_filename)
        
        # FIXED: Create original media name for text overlay (use audio file name)
        original_media_name = audio_base_name.replace("_", " ")
        
        # Process with template
        temp_processed_video = process_concatenated_video_with_template(
            temp_concat_video, template_path, final_output_path, text_settings, original_media_name
        )
        
        if not temp_processed_video:
            print("❌ Template processing failed")
            return False
        
        print(f"📝 Step 3: Adding narasi audio...")
        
        # Add narasi audio
        success = add_narasi_audio_to_video(
            temp_processed_video, audio_path, final_output_path,
            audio_mode, narasi_volume, original_volume
        )
        
        if success:
            print(f"✅ Narasi processing completed: {output_filename}")
            return True
        else:
            print(f"❌ Audio processing failed")
            return False
    
    except Exception as e:
        print(f"❌ Narasi Mode processing error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up temporary files
        temp_files = [temp_concat_video]
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as cleanup_error:
                print(f"⚠️ Could not remove temp file {temp_file}: {cleanup_error}")
        
        print(f"🧹 Cleaned up temporary files")

def create_file_matches(video_files, audio_files):
    """Create matches between video and audio files based on filename."""
    matches = {}
    
    # Get base names (without extensions) for audio files
    audio_base_names = {}
    for audio_file in audio_files:
        base_name = os.path.splitext(audio_file)[0].lower()
        audio_base_names[base_name] = audio_file
    
    # For each audio file, find matching videos
    for base_name, audio_file in audio_base_names.items():
        matched_videos = []
        
        # Find videos that match this audio file name
        for video_file in video_files:
            video_base_name = os.path.splitext(video_file)[0].lower()
            
            # Exact match or partial match
            if (base_name == video_base_name or 
                base_name in video_base_name or 
                video_base_name in base_name):
                matched_videos.append(video_file)
        
        # If no exact matches found, use all videos (fallback)
        if not matched_videos:
            matched_videos = video_files.copy()
        
        matches[audio_file] = matched_videos
    
    return matches

def process_narasi_mode_bulk(video_folder, audio_folder, template_path, output_folder, 
                            text_settings, gpu_settings, audio_mode="narasi_only", 
                            narasi_volume=100, original_volume=30, progress_callback=None):
    """Process narasi mode in bulk with improved matching and processing."""
    print(f"🎬 Starting Narasi Mode bulk processing...")
    print(f"   Video folder: {video_folder}")
    print(f"   Audio folder: {audio_folder}")
    print(f"   Template: {os.path.basename(template_path)}")
    print(f"   Audio mode: {audio_mode}")
    
    try:
        # Get video and audio files
        video_files = get_video_files(video_folder)
        audio_files = get_audio_files(audio_folder)
        
        print(f"📹 Found {len(video_files)} video files")
        print(f"🎵 Found {len(audio_files)} audio files")
        
        if not video_files:
            print("❌ No video files found")
            return False
        
        if not audio_files:
            print("❌ No audio files found")
            return False
        
        # Create file matches
        matches = create_file_matches(video_files, audio_files)
        print(f"🔗 Created {len(matches)} matching pairs")
        
        if not matches:
            print("❌ No matches created")
            return False
        
        # Process each match
        successful_count = 0
        total_matches = len(matches)
        
        for i, (audio_file, matched_videos) in enumerate(matches.items()):
            if progress_callback:
                progress = (i / total_matches) * 100
                progress_callback(progress, f"Processing {i+1}/{total_matches}: {audio_file}")
            
            print(f"\n🎬 Processing match {i+1}/{total_matches}:")
            print(f"   Audio: {audio_file}")
            print(f"   Videos: {len(matched_videos)} files")
            
            audio_base_name = os.path.splitext(audio_file)[0]
            output_filename = f"narasi_{audio_base_name}.mp4"
            print(f"   Output: {output_filename}")
            
            try:
                success = process_single_narasi_match(
                    matched_videos, audio_file, video_folder, audio_folder,
                    template_path, output_folder, text_settings, gpu_settings,
                    audio_mode, narasi_volume, original_volume
                )
                
                if success:
                    successful_count += 1
                    print(f"✅ Completed: {audio_file}")
                else:
                    print(f"❌ Failed: {audio_file}")
            
            except Exception as e:
                print(f"❌ Error processing {audio_file}: {e}")
                continue
        
        print(f"\n🎬 Narasi Mode bulk processing completed!")
        print(f"✅ Successfully processed: {successful_count}/{total_matches} matches")
        print(f"📁 Output folder: {output_folder}")
        
        return successful_count > 0
        
    except Exception as e:
        print(f"❌ Bulk processing error: {e}")
        import traceback
        traceback.print_exc()
        return False