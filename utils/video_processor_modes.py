"""
Video Processor Modes - Processing logic for different modes
"""

import os
import cv2
import numpy as np
import tempfile
from utils.file_operations import (
    get_all_media_files, create_output_folder, add_audio_to_video,
    add_background_music_to_video, add_dual_audio_to_video, is_image_file
)
from utils.green_screen_detection import create_green_screen_mask
from utils.image_processing import process_image_greenscreen, process_image_blur
from utils.narasi_processing import process_narasi_mode_bulk

class VideoProcessorModes:
    """Handles different processing modes."""
    
    def __init__(self, core, gui_manager=None):
        self.core = core
        self.gui_manager = gui_manager
        self.progress_callback = None
    
    def process_greenscreen_mode(self, settings):
        """Process green screen mode."""
        print("🎬 Starting Green Screen Mode processing...")
        
        folder_path = settings['folder_path']
        template_info = settings['template_info']
        text_settings = settings['text_settings']
        audio_settings = settings['audio_settings']
        output_settings = settings['output_settings']
        gpu_settings = settings['gpu_settings']
        
        # Validate inputs
        if not folder_path:
            print("❌ No video folder selected")
            return False
        
        if not template_info or not template_info.get('path'):
            print("❌ No template selected")
            return False
        
        # Get media files
        try:
            media_files = get_all_media_files(folder_path)
            if not media_files:
                print("❌ No media files found")
                return False
            
            print(f"📁 Found {len(media_files)} media files")
        except Exception as e:
            print(f"❌ Error scanning folder: {e}")
            return False
        
        # Create output folder
        if output_settings['custom_enabled'] and output_settings['custom_folder']:
            output_folder = output_settings['custom_folder']
        else:
            output_folder = create_output_folder(folder_path, "edited_videos_greenscreen")
        
        print(f"📁 Output folder: {output_folder}")
        
        # Load template
        template_path = template_info['path']
        
        # Check if template is video
        if template_info.get('is_video', False):
            return self.process_with_video_template(
                media_files, folder_path, template_info, output_folder,
                text_settings, audio_settings, gpu_settings
            )
        else:
            # Static image template
            template = cv2.imread(template_path)
            if template is None:
                print("❌ Could not load template")
                return False
            
            template = cv2.resize(template, (1080, 1920))
            template_mask = create_green_screen_mask(template)
            
            if np.sum(template_mask) == 0:
                print("❌ No green screen detected in template")
                return False
            
            return self.process_with_static_template(
                media_files, folder_path, template, template_mask, output_folder,
                text_settings, audio_settings, gpu_settings
            )
    
    def process_with_video_template(self, media_files, folder_path, template_info, 
                                  output_folder, text_settings, audio_settings, gpu_settings):
        """Process media files with video template."""
        print("🎬 Processing with video template...")
        
        template_path = template_info['path']
        successful_count = 0
        total_files = len(media_files)
        
        for i, media_file in enumerate(media_files):
            if self.progress_callback:
                progress = (i / total_files) * 100
                self.progress_callback(progress, f"Processing {i+1}/{total_files}: {media_file}")
            
            try:
                media_path = os.path.join(folder_path, media_file)
                output_path = os.path.join(output_folder, f"greenscreen_{media_file}")
                
                # Ensure output is MP4
                if not output_path.lower().endswith('.mp4'):
                    output_path = os.path.splitext(output_path)[0] + '.mp4'
                
                print(f"\n🎬 Processing {i+1}/{total_files}: {media_file}")
                
                # Check if input is image
                if is_image_file(media_path):
                    # FIXED: Use new function for image with video template (duration follows template)
                    from utils.image_processing import process_image_with_video_template_duration
                    temp_output = process_image_with_video_template_duration(
                        media_path, template_info, output_path, text_settings
                    )
                    
                    # FIXED: Images also need audio processing for background music
                    if temp_output:
                        # Create a dummy video path for audio processing (since image doesn't have original audio)
                        success = self.add_audio_to_processed_video(
                            temp_output, media_path, audio_settings, final_output_path=output_path
                        )
                    else:
                        success = False
                else:
                    # Video input with video template
                    temp_output = self.process_video_with_video_template(
                        media_path, template_path, output_path, text_settings, gpu_settings
                    )
                    
                    if temp_output:
                        # Add audio from input video (not template)
                        success = self.add_audio_to_processed_video(
                            temp_output, media_path, audio_settings, final_output_path=output_path
                        )
                    else:
                        success = False
                
                if success:
                    successful_count += 1
                    print(f"✅ Completed: {media_file}")
                else:
                    print(f"❌ Processing failed: {media_file}")
            
            except Exception as e:
                print(f"❌ Error processing {media_file}: {e}")
                continue
        
        print(f"\n🎬 Green Screen Mode completed!")
        print(f"✅ Successfully processed: {successful_count}/{total_files} files")
        
        return successful_count > 0
    
    def process_video_with_video_template(self, video_path, template_path, output_path, 
                                        text_settings, gpu_settings):
        """Process video with video template - creates MP4 output."""
        from utils.video_template_processing import process_video_with_video_template
        
        # FIXED: Get temp file path and handle audio processing
        temp_output = process_video_with_video_template(
            template_path, video_path, output_path, text_settings
        )
        
        if temp_output and isinstance(temp_output, str):
            # Process audio using input video (not template)
            print(f"🎵 Processing audio from input video (template muted)")
            return temp_output  # Return temp file for audio processing
        else:
            return False
    
    def process_image_with_video_template(self, image_path, template_path, output_path, 
                                        text_settings, gpu_settings):
        """Process image with video template - creates MP4 output."""
        from utils.video_template_processing import process_image_with_video_template
        
        # FIXED: Handle image with video template (no audio needed)
        success = process_image_with_video_template(
            template_path, image_path, output_path, text_settings
        )
        
        return success
    
    def process_with_static_template(self, media_files, folder_path, template, template_mask,
                                   output_folder, text_settings, audio_settings, gpu_settings):
        """Process media files with static template."""
        print("🖼️ Processing with static template...")
        
        successful_count = 0
        total_files = len(media_files)
        
        for i, media_file in enumerate(media_files):
            if self.progress_callback:
                progress = (i / total_files) * 100
                self.progress_callback(progress, f"Processing {i+1}/{total_files}: {media_file}")
            
            try:
                media_path = os.path.join(folder_path, media_file)
                output_path = os.path.join(output_folder, f"greenscreen_{media_file}")
                
                # Ensure output is MP4
                if not output_path.lower().endswith('.mp4'):
                    output_path = os.path.splitext(output_path)[0] + '.mp4'
                
                print(f"\n🎬 Processing {i+1}/{total_files}: {media_file}")
                
                # Check file type and process accordingly
                if is_image_file(media_path):
                    # FIXED: Get temp output for audio processing
                    temp_output = process_image_greenscreen(
                        media_path, template, template_mask, output_path, text_settings
                    )
                    
                    # FIXED: Process audio for images too
                    if temp_output:
                        success = self.add_audio_to_processed_video(
                            temp_output, media_path, audio_settings, final_output_path=output_path
                        )
                    else:
                        success = False
                else:
                    # Video processing
                    temp_output = self.core.process_single_video(
                        media_path, template, template_mask, output_path, 
                        text_settings, gpu_settings
                    )
                    
                    if temp_output:
                        success = self.add_audio_to_processed_video(
                            temp_output, media_path, audio_settings, final_output_path=output_path
                        )
                    else:
                        success = False
                
                if success:
                    successful_count += 1
                    print(f"✅ Completed: {media_file}")
                else:
                    print(f"❌ Processing failed: {media_file}")
            
            except Exception as e:
                print(f"❌ Error processing {media_file}: {e}")
                continue
        
        print(f"\n🎬 Green Screen Mode completed!")
        print(f"✅ Successfully processed: {successful_count}/{total_files} files")
        
        return successful_count > 0
    
    def process_blur_mode(self, settings):
        """Process blur mode."""
        print("🌀 Starting Blur Mode processing...")
        
        folder_path = settings['folder_path']
        blur_settings = settings['blur_settings']
        text_settings = settings['text_settings']
        audio_settings = settings['audio_settings']
        output_settings = settings['output_settings']
        gpu_settings = settings['gpu_settings']
        
        # Validate inputs
        if not folder_path:
            print("❌ No video folder selected")
            return False
        
        # Get media files
        try:
            media_files = get_all_media_files(folder_path)
            if not media_files:
                print("❌ No media files found")
                return False
            
            print(f"📁 Found {len(media_files)} media files")
        except Exception as e:
            print(f"❌ Error scanning folder: {e}")
            return False
        
        # Create output folder
        if output_settings['custom_enabled'] and output_settings['custom_folder']:
            output_folder = output_settings['custom_folder']
        else:
            output_folder = create_output_folder(folder_path, "edited_videos_blur")
        
        print(f"📁 Output folder: {output_folder}")
        
        successful_count = 0
        total_files = len(media_files)
        
        for i, media_file in enumerate(media_files):
            if self.progress_callback:
                progress = (i / total_files) * 100
                self.progress_callback(progress, f"Processing {i+1}/{total_files}: {media_file}")
            
            try:
                media_path = os.path.join(folder_path, media_file)
                output_path = os.path.join(output_folder, f"blur_{media_file}")
                
                # Ensure output is MP4
                if not output_path.lower().endswith('.mp4'):
                    output_path = os.path.splitext(output_path)[0] + '.mp4'
                
                print(f"\n🌀 Processing {i+1}/{total_files}: {media_file}")
                
                # Check file type and process accordingly
                if is_image_file(media_path):
                    # FIXED: Get temp output for audio processing
                    temp_output = process_image_blur(
                        media_path, output_path, blur_settings, text_settings
                    )
                    
                    # FIXED: Process audio for images too
                    if temp_output:
                        success = self.add_audio_to_processed_video(
                            temp_output, media_path, audio_settings, output_path
                        )
                    else:
                        success = False
                else:
                    # Video processing
                    temp_output = self.core.process_single_video_blur(
                        media_path, output_path, blur_settings, text_settings, gpu_settings
                    )
                    
                    if temp_output:
                        success = self.add_audio_to_processed_video(
                            temp_output, media_path, audio_settings, output_path
                        )
                    else:
                        success = False
                
                if success:
                    successful_count += 1
                    print(f"✅ Completed: {media_file}")
                else:
                    print(f"❌ Processing failed: {media_file}")
            
            except Exception as e:
                print(f"❌ Error processing {media_file}: {e}")
                continue
        
        print(f"\n🌀 Blur Mode completed!")
        print(f"✅ Successfully processed: {successful_count}/{total_files} files")
        
        return successful_count > 0
    
    def process_narasi_mode(self, settings):
        """Process narasi mode."""
        print("🎙️ Starting Narasi Mode processing...")
        
        folder_path = settings['folder_path']
        template_info = settings['template_info']
        narasi_settings = settings['narasi_settings']
        text_settings = settings['text_settings']
        output_settings = settings['output_settings']
        gpu_settings = settings['gpu_settings']
        
        # Validate inputs
        if not folder_path:
            print("❌ No video folder selected")
            return False
        
        if not template_info or not template_info.get('path'):
            print("❌ No template selected")
            return False
        
        if not narasi_settings.get('audio_folder_path'):
            print("❌ No audio folder selected")
            return False
        
        # Create output folder
        if output_settings['custom_enabled'] and output_settings['custom_folder']:
            output_folder = output_settings['custom_folder']
        else:
            output_folder = create_output_folder(folder_path, "edited_videos_narasi")
        
        print(f"📁 Output folder: {output_folder}")
        
        # Process narasi mode
        return process_narasi_mode_bulk(
            folder_path,
            narasi_settings['audio_folder_path'],
            template_info['path'],
            output_folder,
            text_settings,
            gpu_settings,
            narasi_settings.get('audio_mode', 'narasi_only'),
            narasi_settings.get('narasi_volume', 100),
            narasi_settings.get('original_volume', 30),
            self.progress_callback
        )
    
    def add_audio_to_processed_video(self, temp_video_path, original_video_path, 
                                   audio_settings, final_output_path=None):
        """Add audio to processed video based on settings."""
        if final_output_path is None:
            final_output_path = temp_video_path.replace('_temp.mp4', '.mp4')
        
        audio_mode = audio_settings.get('mode', 'original_only')
        
        print(f"🎵 Audio processing mode: {audio_mode}")
        print(f"   Temp video: {os.path.basename(temp_video_path)}")
        print(f"   Final output: {os.path.basename(final_output_path)}")
        
        try:
            # FIXED: Handle image inputs (no original audio available)
            is_image_input = is_image_file(original_video_path)
            
            if audio_mode == 'original_only':
                if is_image_input:
                    # Images don't have original audio, just move temp file to final output
                    import shutil
                    shutil.move(temp_video_path, final_output_path)
                    print(f"✅ Image processing completed (no original audio)")
                    return True
                else:
                    # Use original audio only
                    return add_audio_to_video(temp_video_path, original_video_path, final_output_path)
            
            elif audio_mode == 'background_only':
                # Use background music only
                audio_folder = audio_settings.get('folder_path', '')
                if not audio_folder:
                    if is_image_input:
                        # No background audio and no original audio, just move file
                        import shutil
                        shutil.move(temp_video_path, final_output_path)
                        print("⚠️ No background audio folder, image output without audio")
                        return True
                    else:
                        print("⚠️ No background audio folder, using original audio")
                        return add_audio_to_video(temp_video_path, original_video_path, final_output_path)
                
                # FIXED: Check if audio folder exists and has files
                if not os.path.exists(audio_folder):
                    print(f"❌ Audio folder does not exist: {audio_folder}")
                    if is_image_input:
                        import shutil
                        shutil.move(temp_video_path, final_output_path)
                        print("❌ Audio folder missing, image output without audio")
                        return True
                    else:
                        return add_audio_to_video(temp_video_path, original_video_path, final_output_path)
                
                # Select random audio file
                from utils.file_operations import get_audio_files
                import random
                
                audio_files = get_audio_files(audio_folder)
                if not audio_files:
                    print(f"❌ No audio files found in folder: {audio_folder}")
                    if is_image_input:
                        import shutil
                        shutil.move(temp_video_path, final_output_path)
                        print("❌ No audio files found, image output without audio")
                        return True
                    else:
                        return add_audio_to_video(temp_video_path, original_video_path, final_output_path)
                
                selected_audio = random.choice(audio_files)
                background_audio_path = os.path.join(audio_folder, selected_audio)
                background_volume = audio_settings.get('background_volume', 50)
                
                print(f"🎵 Selected background audio: {selected_audio}")
                
                # FIXED: For images, create a dummy original video path for background music processing
                if is_image_input:
                    # Use temp video as "original" since image has no audio
                    return add_background_music_to_video(
                        temp_video_path, temp_video_path, background_audio_path,
                        final_output_path, background_volume
                    )
                else:
                    return add_background_music_to_video(
                        temp_video_path, original_video_path, background_audio_path,
                        final_output_path, background_volume
                    )
            
            elif audio_mode == 'dual_mixing':
                if is_image_input:
                    print("⚠️ Dual mixing not available for images, using background only")
                    # Fall back to background only for images
                    audio_folder = audio_settings.get('folder_path', '')
                    if audio_folder and os.path.exists(audio_folder):
                        from utils.file_operations import get_audio_files
                        import random
                        
                        audio_files = get_audio_files(audio_folder)
                        if audio_files:
                            selected_audio = random.choice(audio_files)
                            background_audio_path = os.path.join(audio_folder, selected_audio)
                            background_volume = audio_settings.get('background_volume', 50)
                            
                            return add_background_music_to_video(
                                temp_video_path, temp_video_path, background_audio_path,
                                final_output_path, background_volume
                            )
                    
                    # No background audio available, just move file
                    import shutil
                    shutil.move(temp_video_path, final_output_path)
                    return True
                
                # Mix original + background audio
                audio_folder = audio_settings.get('folder_path', '')
                if not audio_folder:
                    print("⚠️ No background audio folder, using original audio")
                    return add_audio_to_video(temp_video_path, original_video_path, final_output_path)
                
                # FIXED: Check if audio folder exists and has files
                if not os.path.exists(audio_folder):
                    print(f"❌ Audio folder does not exist: {audio_folder}")
                    return add_audio_to_video(temp_video_path, original_video_path, final_output_path)
                
                # Select random audio file
                from utils.file_operations import get_audio_files
                import random
                
                audio_files = get_audio_files(audio_folder)
                if not audio_files:
                    print(f"❌ No audio files found in folder: {audio_folder}")
                    return add_audio_to_video(temp_video_path, original_video_path, final_output_path)
                
                selected_audio = random.choice(audio_files)
                background_audio_path = os.path.join(audio_folder, selected_audio)
                original_volume = audio_settings.get('original_volume', 100)
                background_volume = audio_settings.get('background_volume', 50)
                
                print(f"🎵 Selected background audio: {selected_audio}")
                
                return add_dual_audio_to_video(
                    temp_video_path, original_video_path, background_audio_path,
                    final_output_path, original_volume, background_volume
                )
            
            else:
                # Default to original audio or no audio for images
                if is_image_input:
                    import shutil
                    shutil.move(temp_video_path, final_output_path)
                    return True
                else:
                    return add_audio_to_video(temp_video_path, original_video_path, final_output_path)
        
        except Exception as e:
            print(f"❌ Audio processing error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to original audio
            try:
                if is_image_file(original_video_path):
                    # For images, just move the temp file
                    import shutil
                    shutil.move(temp_video_path, final_output_path)
                    return True
                else:
                    return add_audio_to_video(temp_video_path, original_video_path, final_output_path)
            except Exception as fallback_error:
                print(f"❌ Fallback audio processing also failed: {fallback_error}")
                return False