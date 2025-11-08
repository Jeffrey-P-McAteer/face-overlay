use clap::{Parser, ValueEnum};

#[derive(Parser, Debug)]
#[command(name = "face-overlay")]
#[command(about = "A webcam face overlay with background segmentation for Wayland")]
#[command(long_about = "Captures webcam feed, performs AI-based foreground/background segmentation, and displays the result as a transparent overlay on the desktop using wlr-layer-shell protocol.")]
pub struct Args {
    #[arg(
        short = 'd',
        long = "device",
        default_value = "/dev/video0",
        help = "Video device path"
    )]
    pub device: String,

    #[arg(
        short = 'o',
        long = "output",
        default_value = "/dev/null",
        help = "Output Video File"
    )]
    pub output: String,

    #[arg(
        long = "input-audio-mic-device",
        default_value = "alsa_input.usb-C-Media_Electronics_Inc._TONOR_TC-777_Audio_Device-00.pro-input-0",
        help = "Input alsa device name for mic input (output from 'pw-cli list-objects Node | grep -i node.name')"
    )]
    pub input_audio_mic_device: String,

    #[arg(
        long = "input-audio-mic-volume",
        default_value = "2.5",
        help = "Mic input volume, values above 1.0 are a boost. Default is 2.5"
    )]
    pub input_audio_mic_volume: String,

    #[arg(
        long = "input-audio-noise-reduce-amount",
        default_value = "0.4",
        help = "Noise reduction argument passed to sox during post-process of the mic recording."
    )]
    pub input_audio_noise_reduce_amount: String,

    #[arg(
        long = "record-noise-profile",
        default_value = "0",
        help = "Records N seconds of sound from --input-audio-mic-device and creates a noise profile from is using sox, output to --mic-noise-profile"
    )]
    pub action_record_input_audio_mic_noise_profile: usize,
    #[arg(
        long = "mic-noise-profile",
        default_value = "/j/.cache/face-overlay-sox-noise-profile.noise",
        help = "Mic noise profile created by sox. Pass --record-noise-profile to create using microphone input."
    )]
    pub input_audio_mic_noise_profile: String,

    #[arg(
        long = "input-audio-monitor-device",
        default_value = "bluez_output.F4_9D_8A_D2_E3_01.1",
        help = "Input alsa device name for apps input (output from 'pw-cli list-objects Node | grep -i node.name')"
    )]
    pub input_audio_monitor_device: String,


    #[arg(
        long = "input-events",
        default_value = "/dev/input/by-id/usb-Griffin_Technology__Inc._Griffin_PowerMate-event-if00",
        help = "Input event file for screen capture zoom"
    )]
    pub input_events_file: String,

    #[arg(
        short = 'a',
        long = "anchor",
        default_value = "lower-right",
        help = "Anchor position for the overlay"
    )]
    pub anchor: AnchorPosition,

    #[arg(
        short = 'm',
        long = "model",
        help = "Path to the ONNX segmentation model file"
    )]
    pub model_path: Option<String>,

    #[arg(
        long = "hf-token-file",
        help = "Path to file containing Hugging Face access token (for private models)"
    )]
    pub hf_token_file: Option<String>,

    #[arg(
        long = "disable-download",
        help = "Disable automatic model download"
    )]
    pub disable_download: bool,

    #[arg(
        short = 'w',
        long = "width",
        default_value = "640",
        help = "Overlay width in pixels"
    )]
    pub width: u32,

    #[arg(
        long = "height", 
        default_value = "480",
        help = "Overlay height in pixels"
    )]
    pub height: u32,


    #[arg(
        short = 'v',
        long = "verbose",
        action = clap::ArgAction::Count,
        help = "Increase verbosity level (use multiple times for more verbose output)"
    )]
    pub verbose: u8,


    #[arg(
        long = "fps",
        default_value = "30",
        help = "Target frames per second"
    )]
    pub fps: u32,


    #[arg(
        long = "ai-model",
        default_value = "mediapipe-selfie",
        help = "AI model type to use for segmentation (mediapipe-selfie, sinet, u2net, yolov8n-seg, fastsam)"
    )]
    pub ai_model: String,

    #[arg(
        long = "flip",
        default_value = "true",
        help = "Flip the camera input horizontally (mirror effect)"
    )]
    pub flip: bool,

    #[arg(
        long = "no-flip",
        help = "reverse of --flip"
    )]
    pub no_flip: bool,

    #[arg(
        long = "mask-erosion",
        default_value = "1",
        help = "Erode (clip inwards) the AI mask by this many pixels (0-5)"
    )]
    pub mask_erosion: u8,

    #[arg(
        long = "border-width",
        default_value = "1",
        help = "Width of the border outline around visible pixels (0 = no border)"
    )]
    pub border_width: u32,

    #[arg(
        long = "border-color",
        default_value = "#101010",
        help = "Color of the border outline in #RRGGBB format"
    )]
    pub border_color: String,

    // Removed ai_inference_interval - AI now runs on EVERY frame for maximum responsiveness
    // Removed mask_cache_size - no more inefficient caching system

}

#[derive(ValueEnum, Clone, Debug)]
pub enum AnchorPosition {
    #[value(name = "lower-left")]
    LowerLeft,
    #[value(name = "lower-right")]
    LowerRight,
}

impl From<AnchorPosition> for crate::wayland_overlay::AnchorPosition {
    fn from(pos: AnchorPosition) -> Self {
        match pos {
            AnchorPosition::LowerLeft => crate::wayland_overlay::AnchorPosition::LowerLeft,
            AnchorPosition::LowerRight => crate::wayland_overlay::AnchorPosition::LowerRight,
        }
    }
}

impl Args {
    pub fn parse_args() -> Self {
        Self::parse()
    }

    pub fn get_ai_model_type(&self) -> crate::segmentation::ModelType {
        match self.ai_model.as_str() {
            "mediapipe-selfie" | "mediapipe" | "selfie" => crate::segmentation::ModelType::MediaPipeSelfie,
            "sinet" => crate::segmentation::ModelType::SINet,
            "u2net" => crate::segmentation::ModelType::U2Net,
            "yolov8n-seg" | "yolo" => crate::segmentation::ModelType::YoloV8nSeg,
            "fastsam" => crate::segmentation::ModelType::FastSam,
            _ => {
                eprintln!("Warning: Unknown model type '{}', defaulting to MediaPipe Selfie", self.ai_model);
                crate::segmentation::ModelType::MediaPipeSelfie
            }
        }
    }

    pub fn get_model_path(&self) -> std::path::PathBuf {
        match &self.model_path {
            Some(path) => std::path::PathBuf::from(path),
            None => {
                let default_path = std::path::PathBuf::from("models/u2net.onnx");
                if !default_path.exists() {
                    eprintln!("Warning: Default model path {:?} does not exist.", default_path);
                    eprintln!("Please specify a model path with --model or place a model at the default location.");
                }
                default_path
            }
        }
    }

    pub fn get_border_color(&self) -> [u8; 3] {
        if let Some(hex_str) = self.border_color.strip_prefix('#') {
            if hex_str.len() == 6 {
                if let Ok(r) = u8::from_str_radix(&hex_str[0..2], 16) {
                    if let Ok(g) = u8::from_str_radix(&hex_str[2..4], 16) {
                        if let Ok(b) = u8::from_str_radix(&hex_str[4..6], 16) {
                            return [r, g, b];
                        }
                    }
                }
            }
        }
        eprintln!("Warning: Invalid border color '{}', defaulting to white", self.border_color);
        [255, 255, 255]
    }

    pub fn setup_logging(&self) {
        use tracing_subscriber::{EnvFilter, FmtSubscriber};

        let level = match self.verbose {
            0 => "warn",
            1 => "info", 
            2 => "debug",
            //_ => "trace", // Trace is _way_ too noisy
            _ => "debug",
        };

        let subscriber = FmtSubscriber::builder()
            .with_env_filter(
                EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| EnvFilter::new(level))
            )
            .with_target(false)
            .with_thread_ids(false)
            .with_file(true)
            .with_line_number(true)
            .finish();

        tracing::subscriber::set_global_default(subscriber)
            .expect("Failed to set tracing subscriber");
    }
}
