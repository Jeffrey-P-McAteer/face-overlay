mod cli;
mod webcam;
mod segmentation;
mod wayland_overlay;
mod mouse_tracker;

use anyhow::{Context, Result};
use cli::Args;
use image::ImageBuffer;
use mouse_tracker::MouseEventHandler;
use segmentation::{download_model_if_needed, read_hf_token_from_file, SegmentationModel, ModelType};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{error, info};
use wayland_overlay::{AnchorPosition, WaylandOverlay};
use webcam::WebcamCapture;

fn load_model(path: &std::path::Path, model_type: ModelType) -> Option<SegmentationModel> {
    if path.exists() {
        if let Ok(model) = SegmentationModel::new(path, model_type.clone()) {
            return Some(model);
        }
    }
    
    let u2net_path = std::env::var("HOME")
        .map(|home| std::path::PathBuf::from(format!("{}/.cache/face-overlay-data/u2net.onnx", home)))
        .unwrap_or_default();
    
    if u2net_path.exists() && model_type != ModelType::U2Net {
        SegmentationModel::new(&u2net_path, ModelType::U2Net).ok()
    } else {
        None
    }
}


#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse_args();
    args.setup_logging();

    info!("Starting face-overlay application");

    let result = run_application(args).await;

    if let Err(e) = result {
        error!("Application error: {:#}", e);
        std::process::exit(1);
    }

    Ok(())
}

async fn run_application(args: Args) -> Result<()> {
    let hf_token = args.hf_token_file.as_ref()
        .and_then(|f| read_hf_token_from_file(f).ok());
    
    let model_path = if let Some(_) = args.model_path {
        args.get_model_path()
    } else {
        download_model_if_needed(hf_token, args.disable_download, Some(args.get_ai_model_type()))
            .await.unwrap_or_default()
    };

    let mut webcam = WebcamCapture::new(Some(&args.device), args.width, args.height)
        .context("Failed to initialize webcam")?;

    let mut segmentation_model = load_model(&model_path, args.get_ai_model_type());

    let anchor_position: AnchorPosition = args.anchor.into();
    let (overlay_width, overlay_height) = webcam.resolution();
    let (mut overlay, _conn, mut event_queue) = WaylandOverlay::new(anchor_position, overlay_width, overlay_height)
        .context("Failed to initialize Wayland overlay")?;

    let qh = event_queue.handle();
    overlay.create_layer_surface(&qh)
        .context("Failed to create layer surface")?;

    let mut mouse_handler = MouseEventHandler::new(args.mouse_flip_delay, !args.disable_mouse_flip)
        .unwrap_or_else(|_| MouseEventHandler::new(args.mouse_flip_delay, false).unwrap());

    let frame_duration = Duration::from_millis(1000 / args.fps as u64);
    let mut last_frame_time = Instant::now();

    loop {
        let now = Instant::now();
        let elapsed = now.duration_since(last_frame_time);

        if elapsed >= frame_duration {
            if let Err(e) = process_frame(&mut webcam, &mut segmentation_model, &mut overlay, &mut mouse_handler, &qh).await {
                error!("Frame processing error: {}", e);
                break;
            }
            last_frame_time = now;
        }

        event_queue.blocking_dispatch(&mut overlay).context("Failed to dispatch Wayland events")?;
        
        let sleep_time = frame_duration.saturating_sub(now.elapsed());
        if sleep_time > Duration::from_millis(1) {
            sleep(sleep_time).await;
        }
    }

    Ok(())
}

async fn process_frame(
    webcam: &mut WebcamCapture,
    segmentation_model: &mut Option<SegmentationModel>,
    overlay: &mut WaylandOverlay,
    mouse_handler: &mut MouseEventHandler,
    qh: &wayland_client::QueueHandle<WaylandOverlay>,
) -> Result<()> {
    let frame = webcam.capture_frame().context("Failed to capture webcam frame")?;
    let (width, height) = frame.dimensions();

    let processed_frame = match segmentation_model {
        Some(model) => model.segment_foreground(&frame).unwrap_or_else(|_| {
            ImageBuffer::from_fn(width, height, |x, y| {
                let pixel = frame.get_pixel(x, y);
                image::Rgba([pixel[0], pixel[1], pixel[2], 255])
            })
        }),
        None => ImageBuffer::from_fn(width, height, |x, y| {
            let pixel = frame.get_pixel(x, y);
            image::Rgba([pixel[0], pixel[1], pixel[2], 255])
        }),
    };

    if mouse_handler.check_for_flip(overlay.get_surface_bounds(), overlay.get_mouse_position()) {
        let new_anchor = match overlay.get_anchor_position() {
            AnchorPosition::LowerLeft => AnchorPosition::LowerRight,
            AnchorPosition::LowerRight => AnchorPosition::LowerLeft,
        };
        overlay.set_anchor_position(new_anchor, qh);
        mouse_handler.reset_flip_state();
    }

    overlay.update_frame(&processed_frame, qh).context("Failed to update overlay frame")
}
