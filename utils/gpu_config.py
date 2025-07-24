"""
GPU Configuration Module for Video Editor
Handles GPU acceleration settings and codec selection
"""

import cv2
import os
import subprocess
import sys

class GPUConfig:
    def __init__(self):
        # Default GPU settings
        self.USE_GPU = True
        self.GPU_ENCODER = "h264_nvenc"  # Options: "h264_nvenc", "hevc_nvenc", "h264_vaapi", "h264_qsv"
        self.GPU_DECODER = "h264_cuvid"  # Options: "h264_cuvid", "hevc_cuvid", None for CPU decoding
        self.GPU_AVAILABLE = False
        self.SUPPORTED_ENCODERS = []
        self.SUPPORTED_DECODERS = []
        
        # Check GPU availability
        self.check_gpu_support()
    
    def check_gpu_support(self):
        """Check if GPU acceleration is available."""
        try:
            # Check NVIDIA GPU support
            nvidia_available = self.check_nvidia_gpu()
            
            # Check Intel GPU support
            intel_available = self.check_intel_gpu()
            
            # Check AMD GPU support
            amd_available = self.check_amd_gpu()
            
            self.GPU_AVAILABLE = nvidia_available or intel_available or amd_available
            
            if self.GPU_AVAILABLE:
                print("✅ GPU acceleration available")
                self.detect_opencv_support()
            else:
                print("❌ GPU acceleration not available, using CPU")
                self.USE_GPU = False
                
        except Exception as e:
            print(f"⚠️ Error checking GPU support: {e}")
            self.GPU_AVAILABLE = False
            self.USE_GPU = False
    
    def check_nvidia_gpu(self):
        """Check NVIDIA GPU availability."""
        try:
            # Check if nvidia-smi is available
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("🎮 NVIDIA GPU detected")
                self.SUPPORTED_ENCODERS.extend(["h264_nvenc", "hevc_nvenc"])
                self.SUPPORTED_DECODERS.extend(["h264_cuvid", "hevc_cuvid"])
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        return False
    
    def check_intel_gpu(self):
        """Check Intel GPU availability."""
        try:
            # Check for Intel Quick Sync Video support
            if sys.platform.startswith('win'):
                # Windows: Check for Intel GPU
                result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                      capture_output=True, text=True, timeout=5)
                if 'Intel' in result.stdout:
                    print("🔷 Intel GPU detected")
                    self.SUPPORTED_ENCODERS.extend(["h264_qsv", "hevc_qsv"])
                    self.SUPPORTED_DECODERS.extend(["h264_qsv", "hevc_qsv"])
                    return True
            else:
                # Linux: Check for VAAPI support
                if os.path.exists('/dev/dri'):
                    print("🔷 Intel GPU (VAAPI) detected")
                    self.SUPPORTED_ENCODERS.extend(["h264_vaapi", "hevc_vaapi"])
                    return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        return False
    
    def check_amd_gpu(self):
        """Check AMD GPU availability."""
        try:
            if sys.platform.startswith('win'):
                result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                      capture_output=True, text=True, timeout=5)
                if 'AMD' in result.stdout or 'Radeon' in result.stdout:
                    print("🔴 AMD GPU detected")
                    # AMD support is limited, mainly through OpenCL
                    return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        return False
    
    def detect_opencv_support(self):
        """Detect OpenCV GPU support and available backends."""
        try:
            # Check OpenCV build info
            build_info = cv2.getBuildInformation()
            
            # Check for CUDA support
            if "CUDA" in build_info and "YES" in build_info:
                print("🎮 OpenCV CUDA support detected")
            
            # Check for OpenCL support
            if cv2.ocl.haveOpenCL():
                print("🔷 OpenCV OpenCL support detected")
            
            # Test basic GPU functionality
            self.test_gpu_functionality()
            
        except Exception as e:
            print(f"⚠️ Error detecting OpenCV GPU support: {e}")
    
    def test_gpu_functionality(self):
        """Test basic GPU functionality."""
        try:
            # Test if we can create a simple GPU mat
            if cv2.ocl.haveOpenCL():
                cv2.ocl.setUseOpenCL(True)
                print("✅ OpenCL enabled for OpenCV")
            
            print(f"🎬 Supported hardware encoders: {', '.join(self.SUPPORTED_ENCODERS)}")
            print(f"🎞️ Supported hardware decoders: {', '.join(self.SUPPORTED_DECODERS)}")
            
        except Exception as e:
            print(f"⚠️ GPU functionality test failed: {e}")
    
    def get_optimal_encoder(self):
        """Get the best available encoder."""
        if not self.USE_GPU or not self.SUPPORTED_ENCODERS:
            return "libx264"  # CPU fallback
        
        # Priority order: NVENC > QSV > VAAPI
        priority_encoders = ["h264_nvenc", "hevc_nvenc", "h264_qsv", "hevc_qsv", "h264_vaapi", "hevc_vaapi"]
        
        for encoder in priority_encoders:
            if encoder in self.SUPPORTED_ENCODERS:
                return encoder
        
        return "libx264"  # CPU fallback
    
    def get_optimal_decoder(self):
        """Get the best available decoder."""
        if not self.USE_GPU or not self.SUPPORTED_DECODERS:
            return None  # CPU decoding
        
        # Priority order: CUVID > QSV
        priority_decoders = ["h264_cuvid", "hevc_cuvid", "h264_qsv", "hevc_qsv"]
        
        for decoder in priority_decoders:
            if decoder in self.SUPPORTED_DECODERS:
                return decoder
        
        return None  # CPU fallback
    
    def get_safe_fourcc_codes(self):
        """Get safe fourcc codes that work with most systems."""
        # Return fourcc codes in order of preference
        return [
            ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
            ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
            ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
            ('X264', cv2.VideoWriter_fourcc(*'X264')),
        ]
    
    def get_encoder_params(self, encoder):
        """Get optimal parameters for specific encoder."""
        params = {}
        
        if "nvenc" in encoder:
            # NVIDIA NVENC parameters
            params = {
                'preset': 'fast',
                'tune': 'hq',
                'rc': 'vbr',
                'cq': '23',
                'b:v': '5M',
                'maxrate': '10M',
                'bufsize': '20M'
            }
        elif "qsv" in encoder:
            # Intel Quick Sync parameters
            params = {
                'preset': 'medium',
                'global_quality': '23',
                'look_ahead': '1',
                'b:v': '5M'
            }
        elif "vaapi" in encoder:
            # VAAPI parameters
            params = {
                'quality': '23',
                'b:v': '5M'
            }
        else:
            # CPU encoder parameters
            params = {
                'preset': 'medium',
                'crf': '23',
                'b:v': '5M'
            }
        
        return params
    
    def set_gpu_enabled(self, enabled):
        """Enable or disable GPU acceleration."""
        self.USE_GPU = enabled and self.GPU_AVAILABLE
        print(f"🎮 GPU acceleration: {'Enabled' if self.USE_GPU else 'Disabled'}")
    
    def set_encoder(self, encoder):
        """Set specific encoder."""
        if encoder in self.SUPPORTED_ENCODERS or encoder == "libx264":
            self.GPU_ENCODER = encoder
            print(f"🎬 Encoder set to: {encoder}")
        else:
            print(f"⚠️ Encoder {encoder} not supported, using default")
    
    def set_decoder(self, decoder):
        """Set specific decoder."""
        if decoder in self.SUPPORTED_DECODERS or decoder is None:
            self.GPU_DECODER = decoder
            print(f"🎞️ Decoder set to: {decoder if decoder else 'CPU'}")
        else:
            print(f"⚠️ Decoder {decoder} not supported, using CPU")
    
    def get_config_summary(self):
        """Get configuration summary."""
        return {
            'gpu_available': self.GPU_AVAILABLE,
            'gpu_enabled': self.USE_GPU,
            'encoder': self.get_optimal_encoder(),
            'decoder': self.get_optimal_decoder(),
            'supported_encoders': self.SUPPORTED_ENCODERS,
            'supported_decoders': self.SUPPORTED_DECODERS
        }

# Global GPU configuration instance
gpu_config = GPUConfig()