# Face Overlay

A Rust application for webcam-based foreground segmentation with transparent overlay using the wlr-layer-shell protocol.

## Features

- ✅ Webcam capture from `/dev/video0` (currently simulated)
- ✅ CLI options for anchor position (lower-left/lower-right)
- ✅ Mouse cursor overlap detection and flip functionality
- ✅ Wayland layer shell integration structure
- 🔄 AI-based foreground/background segmentation (placeholder - requires ONNX model)
- 🔄 Real webcam integration (requires proper V4L2 setup)

## Usage

```bash
# Basic usage with default settings
cargo run

# Specify webcam device and anchor position
cargo run -- --device /dev/video0 --anchor lower-left

# Enable verbose logging
cargo run -- -vv

# Disable mouse flip functionality
cargo run -- --disable-mouse-flip

# Show help
cargo run -- --help
```

## CLI Options

- `-d, --device <DEVICE>`: Video device path (default: `/dev/video0`)
- `-a, --anchor <ANCHOR>`: Anchor position - `lower-left` or `lower-right` (default: `lower-right`)
- `-m, --model <MODEL>`: Path to ONNX segmentation model file
- `-w, --width <WIDTH>`: Overlay width in pixels (default: 640)
- `-h, --height <HEIGHT>`: Overlay height in pixels (default: 480)
- `--mouse-flip-delay <DELAY>`: Delay in milliseconds before flipping overlay when mouse overlaps (default: 1000)
- `-v, --verbose`: Increase verbosity level (use multiple times)
- `--disable-mouse-flip`: Disable automatic flipping when mouse cursor overlaps
- `--fps <FPS>`: Target frames per second (default: 30)

## Architecture

The application consists of several modules:

### `webcam.rs`
- Handles webcam capture from V4L2 devices
- Currently provides simulated video feed for testing
- Will integrate with `nokhwa` crate for real webcam support

### `segmentation.rs`
- AI-based foreground/background segmentation
- Designed to work with ONNX models (U2-Net, etc.)
- Currently disabled - will require model file and `ort` crate

### `wayland_overlay.rs`
- Wayland layer shell protocol implementation
- Creates transparent overlay windows
- Supports anchor positioning and dynamic repositioning

### `mouse_tracker.rs`
- Detects mouse cursor overlap with overlay
- Implements flip delay mechanism
- Supports both global and window-relative positioning

### `cli.rs`
- Command-line argument parsing with `clap`
- Logging configuration
- Application configuration management

## Building

```bash
# Check code
cargo check

# Build release version
cargo build --release

# Run with logging
RUST_LOG=debug cargo run
```

## Dependencies

Core dependencies include:
- `image`: Image processing and format support
- `smithay-client-toolkit`: Wayland client toolkit
- `clap`: Command-line argument parsing
- `tokio`: Async runtime
- `tracing`: Structured logging

Optional dependencies (currently disabled):
- `nokhwa`: Cross-platform webcam capture
- `ort`: ONNX Runtime for ML inference

## Development Status

This is a proof-of-concept implementation. Current limitations:

1. **Simulated webcam**: Real V4L2 integration requires additional setup
2. **No AI segmentation**: Requires ONNX model file and proper ML dependencies
3. **Basic mouse tracking**: Uses simplified position detection
4. **Wayland-only**: Designed specifically for wlr-layer-shell protocol

## Future Enhancements

- [ ] Real webcam capture with `nokhwa`
- [ ] AI segmentation with downloadable models
- [ ] GPU acceleration for real-time processing
- [ ] Configuration file support
- [ ] Performance optimizations
- [ ] Cross-platform support (X11 fallback)

## License

This project demonstrates defensive security concepts and computer vision techniques for educational purposes.