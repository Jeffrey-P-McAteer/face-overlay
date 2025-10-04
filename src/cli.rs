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
        long = "mouse-flip-delay",
        default_value = "1000",
        help = "Delay in milliseconds before flipping overlay when mouse overlaps"
    )]
    pub mouse_flip_delay: u64,

    #[arg(
        short = 'v',
        long = "verbose",
        action = clap::ArgAction::Count,
        help = "Increase verbosity level (use multiple times for more verbose output)"
    )]
    pub verbose: u8,

    #[arg(
        long = "disable-mouse-flip",
        help = "Disable automatic flipping when mouse cursor overlaps"
    )]
    pub disable_mouse_flip: bool,

    #[arg(
        long = "fps",
        default_value = "30",
        help = "Target frames per second"
    )]
    pub fps: u32,
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

    pub fn setup_logging(&self) {
        use tracing_subscriber::{EnvFilter, FmtSubscriber};

        let level = match self.verbose {
            0 => "warn",
            1 => "info", 
            2 => "debug",
            _ => "trace",
        };

        let subscriber = FmtSubscriber::builder()
            .with_env_filter(
                EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| EnvFilter::new(level))
            )
            .with_target(false)
            .with_thread_ids(false)
            .with_file(false)
            .with_line_number(false)
            .finish();

        tracing::subscriber::set_global_default(subscriber)
            .expect("Failed to set tracing subscriber");
    }
}