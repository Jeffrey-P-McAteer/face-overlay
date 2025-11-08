
#![allow(unused_variables, unused_mut, dead_code)]

mod cli;
mod webcam;
mod segmentation;
mod wayland_overlay;

use image::Rgb;
use anyhow::{Context, Result};
use cli::Args;
use image::ImageBuffer;
use segmentation::{download_model_if_needed, read_hf_token_from_file, SegmentationModel, ModelType};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{error, info};
use std::collections::VecDeque;
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


#[tokio::main(flavor = "multi_thread", worker_threads = 6)]
async fn main() -> Result<()> {
    let mut args = Args::parse_args();
    args.setup_logging();

    info!("Starting face-overlay application");
    if args.no_flip {
        args.flip = false; // args.no_flip overrides flip
    }
    if args.flip {
        info!("Camera input will be flipped horizontally (mirror effect)");
    }

    let result = run_application(args).await;

    if let Err(e) = result {
        error!("Application error: {:#}", e);
        std::process::exit(1);
    }

    Ok(())
}

fn spawn_webcam_capture_task(
    device: String,
    width: u32,
    height: u32,
    flip: bool,
    tx_camera_frame: tokio::sync::watch::Sender<Option<ImageBuffer<Rgb<u8>, Vec<u8>>>>,
    cancel_bool: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> tokio::task::JoinHandle<Result<(), anyhow::Error>> {
    tokio::spawn(async move {
        let mut webcam = WebcamCapture::new(Some(&device), width, height, flip)
            .context("Failed to initialize webcam")?;
        let mut allowed_errors = 10;
        while allowed_errors > 0 && !cancel_bool.load(std::sync::atomic::Ordering::Relaxed) {
            match webcam.capture_frame().context("Failed to capture webcam frame") {
                Ok(frame) => {
                    if let Err(e) = tx_camera_frame.send(Some(frame)) {
                        eprintln!("{:?}", e);
                    }
                }
                Err(e) => {
                    eprintln!("{:?}", e);
                    allowed_errors -= 1;
                }
            }
        }
        Ok(())
    })
}

fn spawn_frame_slicing_task(
    mut rx_camera_frame: tokio::sync::watch::Receiver<Option<ImageBuffer<Rgb<u8>, Vec<u8>>>>,
    mut rx_ai_mask_data: tokio::sync::watch::Receiver<Option<ImageBuffer<image::Luma<u8>, Vec<u8>>>>,
    tx_sliced_frame: tokio::sync::watch::Sender<Option<ImageBuffer<image::Rgba<u8>, Vec<u8>>>>,
    border_width: u32,
    border_color: [u8; 3],
    cancel_bool: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> tokio::task::JoinHandle<Result<(), anyhow::Error>> {
    tokio::spawn(async move {
        let mut allowed_errors = 10;
        while allowed_errors > 0 && !cancel_bool.load(std::sync::atomic::Ordering::Relaxed) {
            // Grab the latest from the webcam
            let frame_value = (*rx_camera_frame.borrow()).clone();
            rx_camera_frame.mark_unchanged(); // Say we have observed the current value

            // Grab the latest AI mask
            let ai_mask = (*rx_ai_mask_data.borrow()).clone();
            rx_ai_mask_data.mark_unchanged(); // Say we have observed the current value

            if let (Some(frame), Some(mask)) = (frame_value, ai_mask) {
                // Slice it and send to the screen! (this should be FAST)
                match SegmentationModel::apply_mask_with_border(&frame, &mask, border_width, border_color) {
                    Ok(frame_a) => {
                        // send
                        if let Err(e) = tx_sliced_frame.send(Some(frame_a)) {
                            eprintln!("[ tx_sliced_frame.send ] {:?}", e);
                        }
                    }
                    Err(e) => {
                        error!("[ SegmentationModel::apply_mask_with_border ] {:?}", e);
                    }
                }
            }
        }
        Ok(())
    })
}

fn spawn_ai_mask_generation_task(
    mut rx_camera_frame: tokio::sync::watch::Receiver<Option<ImageBuffer<Rgb<u8>, Vec<u8>>>>,
    segmentation_model: Option<SegmentationModel>,
    tx_ai_mask_data: tokio::sync::watch::Sender<Option<ImageBuffer<image::Luma<u8>, Vec<u8>>>>,
    mask_erosion: u8,
    cancel_bool: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> tokio::task::JoinHandle<Result<(), anyhow::Error>> {
    tokio::spawn(async move {
        let mut segmentation_model = segmentation_model.expect("We have no segmentation_model!");
        let mut allowed_errors = 10;
        while allowed_errors > 0 && !cancel_bool.load(std::sync::atomic::Ordering::Relaxed) {
            // Grab the latest from the webcam
            let frame_value = (*rx_camera_frame.borrow()).clone();
            rx_camera_frame.mark_unchanged(); // Say we have observed the current value
            if let Some(frame) = frame_value {
                // Slice image and update meta-data for slice_task
                let resized_image = image::imageops::resize(&frame, segmentation_model.input_width as u32, segmentation_model.input_height as u32, image::imageops::FilterType::Nearest);

                let mut mask = segmentation_model.run_efficient_ai_inference(&resized_image)?;
                
                // Apply mask erosion if requested
                if mask_erosion > 0 {
                    mask = SegmentationModel::erode_mask(&mask, mask_erosion)?;
                }

                // send slice data; this is async to the entire imaging pipeline
                if let Err(e) = tx_ai_mask_data.send(Some(mask)) {
                   eprintln!("[ tx_ai_mask_data.send ] {:?}", e);
                }
            }
        }
        Ok(())
    })
}

pub static ZOOM_AMOUNT: atomic_float::AtomicF32 = atomic_float::AtomicF32::new(0.0);

fn spawn_zoom_input_reader(
    cancel_bool: std::sync::Arc<std::sync::atomic::AtomicBool>,
    input_event_file: String,
    verbose: u8,
) -> tokio::task::JoinHandle<Result<(), anyhow::Error>> {
    use evdev::Device;
    tokio::task::spawn_blocking(move || {
        let mut allowed_errors = 10;
        while allowed_errors > 0 && !cancel_bool.load(std::sync::atomic::Ordering::Relaxed) {
            match Device::open(&input_event_file) {
                Ok(mut device) => {
                    if let Err(e) = device.set_nonblocking(true) {
                        eprintln!("{}:{} {:?}", file!(), line!(), e);
                    }
                    while allowed_errors > 0 && !cancel_bool.load(std::sync::atomic::Ordering::Relaxed) {
                        match device.fetch_events() {
                            Ok(event_stream) => {
                                for event in event_stream {
                                    if verbose > 2 {
                                        eprintln!("{}:{} {:?}", file!(), line!(), event);
                                    }
                                    if cancel_bool.load(std::sync::atomic::Ordering::Relaxed) {
                                        break;
                                    }
                                    // Process event codes, write to atomic shared data
                                    match event.destructure() {
                                        evdev::EventSummary::RelativeAxis(ev2, evdev::RelativeAxisCode::REL_DIAL, value) => {
                                            let value_f = value as f32;
                                            let current_zoom = ZOOM_AMOUNT.load(core::sync::atomic::Ordering::Acquire);
                                            eprintln!("{}:{} current_zoom={:?}", file!(), line!(), current_zoom);
                                            let new_zoom = current_zoom + value_f;
                                            if new_zoom >= 0.0 && new_zoom <= 10.0 {
                                                ZOOM_AMOUNT.store(new_zoom, core::sync::atomic::Ordering::Release);
                                            }
                                        },
                                        evdev::EventSummary::Key(_event, evdev::KeyCode::BTN_0, 1) => {
                                            eprintln!("{}:{} Knob Pressed", file!(), line!() );
                                        }
                                        _ => { /* NOP */ },
                                    }

                                }
                            }
                            Err(e) => {
                                if e.kind() == std::io::ErrorKind::WouldBlock {
                                    // This is fine, continue to next loop iteration.
                                }
                                else {
                                    allowed_errors -= 1;
                                    eprintln!("{}:{} {:?}", file!(), line!(), e);
                                }
                                if cancel_bool.load(std::sync::atomic::Ordering::Relaxed) {
                                    break;
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    allowed_errors -= 1;
                    eprintln!("{}:{} {:?}", file!(), line!(), e);
                }
            }
        }
        Ok(())
    })
}

fn spawn_screen_recorder(
    cancel_bool: std::sync::Arc<std::sync::atomic::AtomicBool>,
    input_audio_mic_volume: String,
    input_audio_mic_device: String,
    input_audio_monitor_device: String,
    output_mkv_file: String,
    verbose: u8,
) -> tokio::task::JoinHandle<Result<(), anyhow::Error>> {
    tokio::task::spawn(async move {

        let output_mkv_path = std::path::PathBuf::from(&output_mkv_file);

        let output_mkv_folder = output_mkv_path.parent().expect("Must have a parent folder!");

        let output_mkv_name_parts = output_mkv_path.file_name().expect("Must have a name!").to_string_lossy();
        let output_mkv_name_parts = output_mkv_name_parts.split(".").collect::<Vec<_>>();
        let output_mkv_name = output_mkv_name_parts.split_last().unwrap().1.join(".");

        let mic_recording_file = format!("{}/{}.mic.wav", output_mkv_folder.display(), output_mkv_name);
        let monitor_recording_file = format!("{}/{}.monitor.wav", output_mkv_folder.display(), output_mkv_name);

        println!("output_mkv_name = {output_mkv_name}");
        println!("output_mkv_folder = {output_mkv_folder:?}");
        println!("mic_recording_file = {mic_recording_file}");
        println!("monitor_recording_file = {monitor_recording_file}");

        // Depends on wl-screenrec (yay -S wl-screenrec)
        // Get audio devices from 'pactl list short sources'
        let mut recording_child = tokio::process::Command::new("wl-screenrec")
            .args(&[
                "--filename", &output_mkv_file,
                "--bitrate", "5MB", // Default bitrate
            ])
            .spawn()
            .expect("Could not spawn wl-screenrec");

        let mut mic_record_child = tokio::process::Command::new("pw-record")
            .args(&[
                format!("--target={}", input_audio_mic_device),
                "--volume".to_string(), format!("{}", input_audio_mic_volume),
                mic_recording_file.clone(),
            ])
            .spawn()
            .expect("Could not spawn pw-record");

        let mut monitor_record_child = tokio::process::Command::new("pw-record")
            .args(&[
                format!("--target={}", input_audio_monitor_device),
                "-P".to_string(),
                "{ stream.capture.sink=true }".to_string(), // Required for pw-record to open a monitor device
                monitor_recording_file.clone(),
            ])
            .spawn()
            .expect("Could not spawn pw-record");

        loop {
            match recording_child.try_wait()? {
                Some(status) => {
                    println!("wl-screenrec exited with: {}", status);
                    let _ = mic_record_child.start_kill();
                    let _ = monitor_record_child.start_kill();
                    break;
                }
                None => {
                    // still recording, just wait
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            }
            if cancel_bool.load(std::sync::atomic::Ordering::Relaxed) {
                // Tell sub-process to exit nicely
                let _ = recording_child.start_kill();
                let _ = mic_record_child.start_kill();
                let _ = monitor_record_child.start_kill();
                // Wait up to 36 seconds, 100ms polls (360 iterations)
                let mut wl_screenrec_clean_exit = false;
                for _ in 0..360 {
                    match recording_child.try_wait()? {
                        Some(status) => {
                            println!("wl-screenrec exited with: {}", status);
                            wl_screenrec_clean_exit = true;
                            break;
                        }
                        None => {
                            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        }
                    }
                }
                if ! wl_screenrec_clean_exit {
                    println!("WARN: wl-screenrec did not exit quickly enough, continuing...");
                }
                break;
            }
        }

        // Just for giggles, in case the above processes still have stray writes
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // When we exit attempt to join the .mp3 recordings and the .mkv file to a single output mkv file w/ both audio streams muxed together
        let output_mp4_file = format!("{}.combine.mp4", &output_mkv_file[0..output_mkv_file.len()-4] );
        let ffmpeg_args: Vec<String> = vec![
            "-y".to_string(), // Overwrite existing outputs
            "-loglevel".to_string(), "error".to_string(),
            "-i".to_string(), output_mkv_file.to_string(),
            "-i".to_string(), mic_recording_file.to_string(),
            "-i".to_string(), monitor_recording_file.to_string(),
            "-filter_complex".to_string(), "[1:a][2:a]amix=inputs=2:duration=longest[aout]".to_string(),
            "-map".to_string(),    "0:v".to_string(),
            "-map".to_string(),    "[aout]".to_string(),
            "-c:v".to_string(),    "libx264".to_string(),
            "-preset".to_string(), "veryfast".to_string(),
            "-crf".to_string(),    "21".to_string(), // lower for higher quality (e.g., 18–20).
            "-c:a".to_string(),    "aac".to_string(),
            "-b:a".to_string(),    "320k".to_string(),
            output_mp4_file,
        ];

        println!("Joining recorded files together with:");
        println!("");
        println!("> ffmpeg {}", ffmpeg_args.join(" "));
        println!("");

        let mut ffmpeg_proc = tokio::process::Command::new("ffmpeg")
            .args(&ffmpeg_args)
            .spawn()
            .expect("Could not spawn ffmpeg");

        if let Err(e) = ffmpeg_proc.wait().await {
            eprintln!("Error waiting for ffmpeg_proc = {:?}", e);
        }

        Ok(())
    })
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

    let mut segmentation_model = load_model(&model_path, args.get_ai_model_type());

    let anchor_position: AnchorPosition = args.anchor.clone().into();
    let (overlay_width, overlay_height) = {
        let webcam = WebcamCapture::new(Some(&args.device), args.width, args.height, args.flip)
            .context("Failed to initialize webcam")?; // Camera briefly open+closed
        webcam.resolution()
    };
    let (mut overlay, _conn, mut event_queue) = WaylandOverlay::new(anchor_position, overlay_width, overlay_height)
        .context("Failed to initialize Wayland overlay")?;

    let qh = event_queue.handle();
    overlay.create_layer_surface(&qh)
        .context("Failed to create layer surface")?;


    let frame_duration = Duration::from_millis(1000 / args.fps as u64);
    let mut last_frame_time = Instant::now();

    // FPS tracking for rolling average of last 20 frames
    let mut frame_times: VecDeque<Instant> = VecDeque::with_capacity(20);
    let mut frame_count = 0u64;
    let mut last_fps_report = Instant::now();

    // Write "true" to signal that threads should exit
    let thread_cancel_bool = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Only retains the most-recent copy of the data
    let (tx_camera_frame, mut rx_camera_frame) = tokio::sync::watch::channel::<Option< ImageBuffer<Rgb<u8>, Vec<u8>> >>(None);

    // Spawn an infinite loop which reads frames from the camera and pushed them into the channel
    let webcam_thread_cancel_bool = thread_cancel_bool.clone();
    let webcam_task = spawn_webcam_capture_task(
        args.device.clone(),
        args.width,
        args.height,
        args.flip,
        tx_camera_frame,
        webcam_thread_cancel_bool,
    );

    // The slicer_task needs to see this data
    let (tx_ai_mask_data, mut rx_ai_mask_data) = tokio::sync::watch::channel::<Option< ImageBuffer<image::Luma<u8>, Vec<u8>> >>(None);

    // Spawn an infinite loop which reads frames from the camera and pushed them into the channel
    let (tx_sliced_frame, mut rx_sliced_frame) = tokio::sync::watch::channel::<Option< ImageBuffer<image::Rgba<u8>, Vec<u8>> >>(None);
    let slicer_thread_cancel_bool = thread_cancel_bool.clone();
    let slicer_rx_camera_frame = rx_camera_frame.clone();
    let slicer_task = spawn_frame_slicing_task(
        slicer_rx_camera_frame,
        rx_ai_mask_data.clone(),
        tx_sliced_frame,
        args.border_width,
        args.get_border_color(),
        slicer_thread_cancel_bool,
    );

    // Finally, spawn an infinite AI loop which maintains the pixel mask data to slice against
    let ai_mask_thread_cancel_bool = thread_cancel_bool.clone();
    let ai_mask_rx_camera_frame = rx_camera_frame.clone();
    let ai_mask_task = spawn_ai_mask_generation_task(
        ai_mask_rx_camera_frame,
        segmentation_model,
        tx_ai_mask_data,
        args.mask_erosion,
        ai_mask_thread_cancel_bool,
    );

    let zoom_input_reader_cancel_bool = thread_cancel_bool.clone();
    let zoom_input_reader_task = spawn_zoom_input_reader(
        zoom_input_reader_cancel_bool,
        args.input_events_file,
        args.verbose
    );

    let screen_recorder_cancel_bool = thread_cancel_bool.clone();
    let screen_recorder_task = spawn_screen_recorder(
        screen_recorder_cancel_bool,
        args.input_audio_mic_volume,
        args.input_audio_mic_device,
        args.input_audio_monitor_device,
        args.output,
        args.verbose
    );

    let sigint_cancel_bool = thread_cancel_bool.clone();
    // Spawn a task to handle SIGINT (Ctrl+C)
    tokio::spawn(async move {
        let mut sigint = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt()).expect("Failed to bind SIGINT handler");
        // Wait for signal
        sigint.recv().await;
        println!("SIGINT received, setting sigint_cancel_bool = true");
        sigint_cancel_bool.store(true, std::sync::atomic::Ordering::SeqCst);
    });

    let mut allowed_errors = 10;
    loop {
        let now = Instant::now();
        let elapsed = now.duration_since(last_frame_time);

        if elapsed >= frame_duration {
            // Track frame timing for FPS calculation
            frame_times.push_back(now);
            frame_count += 1;

            // Keep only last 20 frame times
            if frame_times.len() > 20 {
                frame_times.pop_front();
            }

            // Calculate and report FPS every second
            if last_fps_report.elapsed() >= Duration::from_secs(1) {
                let rolling_fps = if frame_times.len() >= 2 {
                    let time_span = frame_times.back().unwrap().duration_since(*frame_times.front().unwrap());
                    if time_span.as_millis() > 0 {
                        (frame_times.len() - 1) as f64 / time_span.as_secs_f64()
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                let shm_stats = overlay.get_shm_stats();
                info!("FPS: {:.1} (rolling avg of last {} frames) | SHM: {} mapping, {}% buffer utilization", 
                      rolling_fps, frame_times.len(),
                      if shm_stats.persistent_mapping_active { "persistent" } else { "standard" },
                      shm_stats.buffer_utilization);
                last_fps_report = now;
            }

            let frame_value = (*rx_sliced_frame.borrow()).clone();
            rx_sliced_frame.mark_unchanged(); // Say we have observed the current value
            if let Some(frame) = frame_value {
                if let Err(e) = process_frame(&frame, &mut overlay, &qh).await {
                    error!("Frame processing error: {}", e);
                    allowed_errors -= 1;
                    if allowed_errors < 1 {
                        break;
                    }
                }
            }

            last_frame_time = now;
        }

        // Async event handling, none of these lines block
        for _ in 0..8 {
            match event_queue.dispatch_pending(&mut overlay).context("Failed to dispatch Wayland events") {
                Ok(_evts) => { }
                Err(e) => {
                    eprintln!("[ event_queue.dispatch_pending ] {:?}", e);
                }
            }
            if let Some(guard) = event_queue.prepare_read() {
                if let Err(e) = guard.read() { // read from socket
                    if !format!("{:?}", e).contains("WouldBlock") {
                        eprintln!("[ event_queue.prepare_read ] {:?}", e);
                    }
                }
            }
            if let Err(e) = event_queue.flush() {
                eprintln!("[ event_queue.flush ] {:?}", e);
            }
        }

        if webcam_task.is_finished() || slicer_task.is_finished() || ai_mask_task.is_finished() || screen_recorder_task.is_finished() {
            break;
        }

        if thread_cancel_bool.load(std::sync::atomic::Ordering::Relaxed) {
            break; // ctrl+c or some other event which means main loop ought to exit
        }

    }


    thread_cancel_bool.store(true, std::sync::atomic::Ordering::Relaxed);
    sleep(Duration::from_millis(110)).await;

    if !webcam_task.is_finished() {
        webcam_task.abort();
    }
    if !slicer_task.is_finished() {
        slicer_task.abort();
    }
    if !ai_mask_task.is_finished() {
        ai_mask_task.abort();
    }

    // Now we wait a bit, reporting status of screen recording
    println!("Waiting for screen_recorder_task (up to 120s)");
    for _ in 0..1200 {
        if screen_recorder_task.is_finished() {
            break;
        }
        sleep(Duration::from_millis(110)).await;
    }
    if !screen_recorder_task.is_finished() {
        println!("WARN: Aborting screen_recorder_task!");
        screen_recorder_task.abort();
    }

    Ok(())
}

async fn process_frame(
    processed_frame: &ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    overlay: &mut WaylandOverlay,
    qh: &wayland_client::QueueHandle<WaylandOverlay>,
) -> Result<()> {

    // Automatically reposition overlay periodically to avoid mouse interference
    overlay.auto_reposition(qh);

    overlay.update_frame(processed_frame, qh).context("Failed to update overlay frame")
}
