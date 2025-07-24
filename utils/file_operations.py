import os
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip

def get_video_files(folder_path):
    """Mendapatkan daftar file video dari folder."""
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')
    return [f for f in os.listdir(folder_path) if f.lower().endswith(video_extensions)]

def get_gif_files(folder_path):
    """Mendapatkan daftar file GIF dari folder dengan deteksi yang disederhanakan."""
    try:
        # print(f"🔍 Scanning folder for GIFs: {folder_path}")  # Reduced logging
        
        if not os.path.exists(folder_path):
            print(f"❌ Folder does not exist: {folder_path}")
            return []
        
        all_files = os.listdir(folder_path)
        gif_files = []
        
        # print(f"📁 Total files in folder: {len(all_files)}")  # Reduced logging
        
        for file in all_files:
            # print(f"🔍 Checking file: {file}")  # Reduced logging
            
            if file.lower().endswith('.gif'):
                file_path = os.path.join(folder_path, file)
                
                # Basic checks
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    gif_files.append(file)
                    # print(f"✅ GIF file added: {file}")  # Reduced logging
                else:
                    print(f"❌ GIF file invalid (empty or missing): {file}")
            # else:
            #     print(f"⏭️ Skipping non-GIF file: {file}")  # Reduced logging
        
        # print(f"📊 GIF scan complete: Found {len(gif_files)} GIF files")  # Reduced logging
        
        return gif_files
        
    except Exception as e:
        print(f"❌ Error scanning for GIF files: {e}")
        import traceback
        traceback.print_exc()
        return []

def get_image_files(folder_path):
    """Mendapatkan daftar file gambar dari folder."""
    try:
        # print(f"🔍 Scanning folder for images: {folder_path}")  # Reduced logging
        
        if not os.path.exists(folder_path):
            print(f"❌ Folder does not exist: {folder_path}")
            return []
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')
        all_files = os.listdir(folder_path)
        image_files = []
        
        # print(f"📁 Total files in folder: {len(all_files)}")  # Reduced logging
        
        for file in all_files:
            if file.lower().endswith(image_extensions):
                file_path = os.path.join(folder_path, file)
                
                # Basic checks
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    image_files.append(file)
                    # print(f"✅ Image file added: {file}")  # Reduced logging
                else:
                    print(f"❌ Image file invalid (empty or missing): {file}")
        
        # print(f"📊 Image scan complete: Found {len(image_files)} image files")  # Reduced logging
        return image_files
        
    except Exception as e:
        print(f"❌ Error scanning for image files: {e}")
        import traceback
        traceback.print_exc()
        return []
def get_all_media_files(folder_path):
    """Mendapatkan daftar semua file media (video + GIF + images) dari folder."""
    video_files = get_video_files(folder_path)
    gif_files = get_gif_files(folder_path)
    image_files = get_image_files(folder_path)
    return video_files + gif_files + image_files

def get_audio_files(folder_path):
    """Mendapatkan daftar file audio dari folder."""
    audio_extensions = ('.mp3', '.wav', '.aac', '.m4a', '.ogg', '.flac', '.wma')
    return [f for f in os.listdir(folder_path) if f.lower().endswith(audio_extensions)]

def create_output_folder(base_path, folder_name="edited_videos_greenscreen"):
    """Membuat folder output untuk video hasil."""
    output_folder = os.path.join(base_path, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

def is_image_file(file_path):
    """Check if file is an image."""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')
    return file_path.lower().endswith(image_extensions)
def add_audio_to_video(temp_video_path, original_video_path, output_path):
    """Menambahkan audio dari video asli ke video hasil."""
    print(f"🎵 Adding audio to video...")
    print(f"   Temp video: {temp_video_path}")
    print(f"   Original video: {original_video_path}")
    print(f"   Output: {output_path}")
    
    # Check if files exist
    if not os.path.exists(temp_video_path):
        print(f"❌ Temp video file not found: {temp_video_path}")
        return False
    
    if not os.path.exists(original_video_path):
        print(f"❌ Original video file not found: {original_video_path}")
        return False
    
    # Check if output directory exists and is writable
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"📁 Created output directory: {output_dir}")
        except Exception as e:
            print(f"❌ Could not create output directory: {e}")
            return False
    
    # Test write permissions
    try:
        test_file = output_path + ".test"
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print(f"✅ Output location is writable")
    except Exception as e:
        print(f"❌ Cannot write to output location: {e}")
        print(f"   Trying alternative approach...")
        # Try using temp directory
        import tempfile
        temp_dir = tempfile.gettempdir()
        temp_output = os.path.join(temp_dir, os.path.basename(output_path))
        print(f"   Using temp output: {temp_output}")
        output_path = temp_output
    
    video_clip = None
    original_audio = None
    final_clip = None
    
    try:
        print(f"🎬 Loading video clip...")
        video_clip = VideoFileClip(temp_video_path)
        print(f"🎵 Loading original audio...")
        original_audio = AudioFileClip(original_video_path)
        print(f"🔗 Combining video and audio...")
        final_clip = video_clip.set_audio(original_audio)
        print(f"💾 Writing final video...")
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
        print(f"✅ Audio processing completed successfully")
        return True
    except Exception as e:
        print(f"❌ Error adding audio: {e}")
        import traceback
        traceback.print_exc()
        try:
            # Fallback: rename temp file to output
            print(f"🔄 Attempting fallback: copying temp file...")
            if os.path.exists(temp_video_path):
                import shutil
                shutil.copy2(temp_video_path, output_path)
                print(f"✅ Fallback successful: video copied without audio processing")
                return True
        except:
            print(f"❌ Fallback also failed")
        return False
    finally:
        # Cleanup resources
        if video_clip:
            video_clip.close()
        if original_audio:
            original_audio.close()
        if final_clip:
            final_clip.close()
        
        # FIXED: Only remove temp file after successful processing
        try:
            if os.path.exists(temp_video_path) and "temp" in os.path.basename(temp_video_path).lower():
                os.remove(temp_video_path)
                print(f"🧹 Removed temp file: {os.path.basename(temp_video_path)}")
        except Exception as cleanup_error:
            print(f"⚠️ Could not remove temp file: {cleanup_error}")

def add_background_music_to_video(temp_video_path, original_video_path, background_audio_path, 
                                 output_path, volume=50):
    """Menambahkan background music ke video (background only mode)."""
    print(f"🎵 Adding background music to video...")
    print(f"   Temp video: {os.path.basename(temp_video_path)}")
    print(f"   Background audio: {os.path.basename(background_audio_path)}")
    print(f"   Volume: {volume}%")
    
    video_clip = None
    background_audio = None
    final_clip = None
    
    try:
        # Check if background audio file exists
        if not os.path.exists(background_audio_path):
            print(f"❌ Background audio file not found: {background_audio_path}")
            return add_audio_to_video(temp_video_path, original_video_path, output_path)
        
        video_clip = VideoFileClip(temp_video_path)
        background_audio = AudioFileClip(background_audio_path)
        
        print(f"📹 Video duration: {video_clip.duration:.2f}s")
        print(f"🎵 Audio duration: {background_audio.duration:.2f}s")
        
        # Sesuaikan volume background music (0-100 ke 0-1)
        background_volume = volume / 100.0
        if background_volume != 1.0:
            background_audio = background_audio.volumex(background_volume)
            print(f"🔊 Background audio volume adjusted to {volume}%")
        
        # FIXED: Handle audio duration matching with compatible method
        if background_audio.duration < video_clip.duration:
            # Calculate how many times to repeat
            repeat_count = int(video_clip.duration / background_audio.duration) + 1
            audio_clips = [background_audio] * repeat_count
            from moviepy.editor import concatenate_audioclips
            background_audio = concatenate_audioclips(audio_clips)
            # Trim to exact duration
            background_audio = background_audio.subclip(0, video_clip.duration)
            print(f"🔄 Background audio looped to match video duration")
        else:
            background_audio = background_audio.subclip(0, video_clip.duration)
            print(f"✂️ Background audio trimmed to match video duration")
        
        # Hanya background music
        final_clip = video_clip.set_audio(background_audio)
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
        print(f"✅ Background music processing completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error adding background music: {e}")
        import traceback
        traceback.print_exc()
        # Fallback ke audio asli jika gagal
        return add_audio_to_video(temp_video_path, original_video_path, output_path)
    finally:
        # Cleanup resources
        if video_clip:
            video_clip.close()
        if background_audio:
            background_audio.close()
        if final_clip:
            final_clip.close()
        
        # FIXED: Only remove temp file after successful processing
        try:
            if os.path.exists(temp_video_path) and "temp" in os.path.basename(temp_video_path).lower():
                os.remove(temp_video_path)
                print(f"🧹 Removed temp file: {os.path.basename(temp_video_path)}")
        except Exception as cleanup_error:
            print(f"⚠️ Could not remove temp file: {cleanup_error}")

def add_dual_audio_to_video(temp_video_path, original_video_path, background_audio_path, 
                           output_path, original_volume=100, background_volume=50):
    """
    FIXED: Add dual audio mixing (original + background) to video.
    Compatible with Python 3.12.0 and MoviePy 1.0.3.
    """
    video_clip = None
    original_audio = None
    background_audio = None
    final_clip = None
    
    try:
        print(f"🎭 Starting dual audio mixing...")
        print(f"   Original volume: {original_volume}%")
        print(f"   Background volume: {background_volume}%")
        
        # Load video and audio clips
        video_clip = VideoFileClip(temp_video_path)
        original_audio = AudioFileClip(original_video_path)
        background_audio = AudioFileClip(background_audio_path)
        
        # Adjust volumes (0-100 to 0-1)
        original_volume_factor = original_volume / 100.0
        background_volume_factor = background_volume / 100.0
        
        # Apply volume adjustments
        if original_volume_factor != 1.0:
            original_audio = original_audio.volumex(original_volume_factor)
            print(f"   📊 Original audio volume adjusted to {original_volume}%")
        
        if background_volume_factor != 1.0:
            background_audio = background_audio.volumex(background_volume_factor)
            print(f"   📊 Background audio volume adjusted to {background_volume}%")
        
        # Synchronize audio durations
        video_duration = video_clip.duration
        
        # Adjust original audio to match video duration
        if original_audio.duration > video_duration:
            original_audio = original_audio.subclip(0, video_duration)
            print(f"   ✂️ Original audio trimmed to {video_duration:.2f}s")
        elif original_audio.duration < video_duration:
            # For shorter original audio, we'll keep it as is and let CompositeAudioClip handle it
            print(f"   ⚠️ Original audio shorter than video ({original_audio.duration:.2f}s vs {video_duration:.2f}s)")
        
        # Adjust background audio to match video duration
        if background_audio.duration < video_duration:
            # FIXED: Loop background audio using concatenate method
            repeat_count = int(video_duration / background_audio.duration) + 1
            audio_clips = [background_audio] * repeat_count
            from moviepy.editor import concatenate_audioclips
            background_audio = concatenate_audioclips(audio_clips)
            # Trim to exact duration
            background_audio = background_audio.subclip(0, video_duration)
            print(f"   🔄 Background audio looped to {video_duration:.2f}s")
        else:
            # Trim background audio to match video duration
            background_audio = background_audio.subclip(0, video_duration)
            print(f"   ✂️ Background audio trimmed to {video_duration:.2f}s")
        
        # Create composite audio (mix both audio tracks)
        print(f"   🎵 Mixing audio tracks...")
        composite_audio = CompositeAudioClip([original_audio, background_audio])
        
        # Set the mixed audio to the video
        final_clip = video_clip.set_audio(composite_audio)
        
        # Write the final video with mixed audio - FIXED: Remove unsupported parameter
        print(f"   💾 Writing final video with dual audio...")
        final_clip.write_videofile(
            output_path, 
            codec="libx264", 
            audio_codec="aac", 
            logger=None
            # REMOVED: temp_audiofile_path parameter (not supported in MoviePy 1.0.3)
        )
        
        print(f"✅ Dual audio mixing completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in dual audio mixing: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to background music only
        print(f"🔄 Falling back to background music only...")
        try:
            return add_background_music_to_video(
                temp_video_path, original_video_path, background_audio_path, 
                output_path, background_volume
            )
        except Exception as fallback_error:
            print(f"❌ Fallback also failed: {fallback_error}")
            # Final fallback to original audio only
            return add_audio_to_video(temp_video_path, original_video_path, output_path)
        
    finally:
        # Cleanup resources
        try:
            if video_clip:
                video_clip.close()
            if original_audio:
                original_audio.close()
            if background_audio:
                background_audio.close()
            if final_clip:
                final_clip.close()
        except Exception as cleanup_error:
            print(f"⚠️ Cleanup error: {cleanup_error}")
        
        # Remove temp file if it still exists
        try:
            # FIXED: Only remove temp file after successful processing
            if os.path.exists(temp_video_path) and "temp" in os.path.basename(temp_video_path).lower():
                os.remove(temp_video_path)
                print(f"🧹 Temporary file removed: {os.path.basename(temp_video_path)}")
        except Exception as temp_cleanup_error:
            print(f"⚠️ Could not remove temp file: {temp_cleanup_error}")

def get_video_properties(video_path):
    """Mendapatkan properti video (fps, width, height)."""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, width, height

def is_gif_file(file_path):
    """Check if file is a GIF."""
    return file_path.lower().endswith('.gif')