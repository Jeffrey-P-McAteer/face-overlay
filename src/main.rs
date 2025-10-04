mod cli;
mod webcam;
mod segmentation;
mod wayland_overlay;
mod mouse_tracker;

use anyhow::{Context, Result};
use cli::Args;
use mouse_tracker::MouseEventHandler;
use segmentation::{download_model_if_needed, SegmentationModel};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{error, info, warn};
use wayland_overlay::{AnchorPosition, WaylandOverlay};
use webcam::WebcamCapture;

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
    let model_path = if args.model_path.is_some() {
        args.get_model_path()
    } else {
        match download_model_if_needed() {
            Ok(path) => path,
            Err(e) => {
                warn!("Model setup failed: {}", e);
                warn!("Continuing without AI segmentation (will show raw webcam feed)");
                std::path::PathBuf::new()
            }
        }
    };

    let mut webcam = WebcamCapture::new(Some(&args.device))
        .context("Failed to initialize webcam")?;

    let segmentation_model = if model_path.exists() {
        match SegmentationModel::new(&model_path) {
            Ok(model) => {
                info!("AI segmentation model loaded successfully");
                Some(model)
            }
            Err(e) => {
                warn!("Failed to load segmentation model: {}", e);
                warn!("Continuing without AI segmentation");
                None
            }
        }
    } else {
        None
    };

    let anchor_position: AnchorPosition = args.anchor.into();
    let (mut overlay, _conn, mut event_queue) = WaylandOverlay::new(anchor_position)
        .context("Failed to initialize Wayland overlay")?;

    let qh = event_queue.handle();
    overlay.create_layer_surface(&qh)
        .context("Failed to create layer surface")?;

    let mut mouse_handler = MouseEventHandler::new(args.mouse_flip_delay, !args.disable_mouse_flip)
        .unwrap_or_else(|e| {
            warn!("Mouse tracking disabled: {}", e);
            MouseEventHandler::new(args.mouse_flip_delay, false).unwrap()
        });

    let frame_duration = Duration::from_millis(1000 / args.fps as u64);
    let mut last_frame_time = Instant::now();

    info!("Starting main loop with {}fps target", args.fps);

    loop {
        let now = Instant::now();
        let elapsed = now.duration_since(last_frame_time);

        if elapsed >= frame_duration {
            if let Err(e) = process_frame(
                &mut webcam,
                &segmentation_model,
                &mut overlay,
                &mut mouse_handler,
                &qh,
            ).await {
                error!("Frame processing error: {}", e);
                break;
            }

            last_frame_time = now;
        }

        event_queue.blocking_dispatch(&mut overlay)
            .context("Failed to dispatch Wayland events")?;

        let sleep_time = frame_duration.saturating_sub(now.elapsed());
        if sleep_time > Duration::from_millis(1) {
            sleep(sleep_time).await;
        }
    }

    info!("Application shutting down");
    Ok(())
}

async fn process_frame(
    webcam: &mut WebcamCapture,
    segmentation_model: &Option<SegmentationModel>,
    overlay: &mut WaylandOverlay,
    mouse_handler: &mut MouseEventHandler,
    qh: &wayland_client::QueueHandle<WaylandOverlay>,
) -> Result<()> {
    let frame = webcam.capture_frame()
        .context("Failed to capture webcam frame")?;

    let processed_frame = if let Some(model) = segmentation_model {
        match model.segment_foreground(&frame) {
            Ok(segmented) => segmented,
            Err(e) => {
                warn!("Segmentation failed: {}, using raw frame", e);
                let (width, height) = frame.dimensions();
                let mut rgba_frame = image::ImageBuffer::new(width, height);
                for (x, y, pixel) in frame.enumerate_pixels() {
                    rgba_frame.put_pixel(x, y, image::Rgba([pixel[0], pixel[1], pixel[2], 255]));
                }
                rgba_frame
            }
        }
    } else {
        let (width, height) = frame.dimensions();
        let mut rgba_frame = image::ImageBuffer::new(width, height);
        for (x, y, pixel) in frame.enumerate_pixels() {
            rgba_frame.put_pixel(x, y, image::Rgba([pixel[0], pixel[1], pixel[2], 255]));
        }
        rgba_frame
    };

    if mouse_handler.check_for_flip(overlay.get_surface_bounds()) {
        let current_anchor = overlay.get_anchor_position();
        let new_anchor = match current_anchor {
            AnchorPosition::LowerLeft => AnchorPosition::LowerRight,
            AnchorPosition::LowerRight => AnchorPosition::LowerLeft,
        };
        overlay.set_anchor_position(new_anchor, qh);
        mouse_handler.reset_flip_state();
        info!("Flipped overlay to opposite side due to mouse overlap");
    }

    overlay.update_frame(&processed_frame, qh)
        .context("Failed to update overlay frame")?;

    Ok(())
}
