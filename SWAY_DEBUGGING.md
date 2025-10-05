# Sway Debugging Guide

If you're not seeing the overlay appear in Sway, this guide will help you troubleshoot the issue.

## Quick Checks

### 1. Verify You're Running Sway
```bash
# Check if Sway is running
echo $XDG_SESSION_TYPE  # Should show "wayland"
echo $WAYLAND_DISPLAY   # Should show something like "wayland-1"
swaymsg -t get_version  # Should show Sway version info
```

### 2. Test with Enhanced Debugging
```bash
# Run with maximum verbosity to see all Wayland protocols
./target/release/face-overlay -vv --debug-wayland

# Check what protocols are available
# You should see "zwlr_layer_shell_v1" in the list
```

### 3. Expected Debug Output
When working correctly, you should see:
```
INFO Wayland environment: WAYLAND_DISPLAY=wayland-1, XDG_SESSION_TYPE=wayland
INFO Available Wayland protocols:
  - wl_compositor v6
  - wl_shm v1
  - wl_output v4
  - zwlr_layer_shell_v1 v4  ← This is crucial for overlays
INFO Successfully connected to Wayland compositor with layer shell support
INFO Creating layer surface with size 640x480, anchor: LowerRight
INFO Layer surface created successfully with anchor: LowerRight
INFO Surface should appear in lower-right corner with 20px margins
```

## Common Issues & Solutions

### Issue 1: "Could not find wayland compositor"
**Symptoms:** Application immediately exits with connection error
**Solution:**
```bash
# Make sure you're in a Wayland session
loginctl show-session $(loginctl | grep $(whoami) | awk '{print $1}') -p Type
# Should show Type=wayland

# If not, log out and select "Sway" from your display manager
```

### Issue 2: "Layer shell not available"
**Symptoms:** Connects to Wayland but fails on layer shell
**Possible causes:**
- Using GNOME/KDE instead of Sway (they don't support wlr-layer-shell)
- Very old Sway version
- Custom compositor that doesn't support layer shell

**Solution:**
```bash
# Check Sway version (layer shell support added in early versions)
swaymsg -t get_version

# Verify layer shell support
wayland-info | grep zwlr_layer_shell_v1
```

### Issue 3: Surface Creates But Nothing Visible
**Symptoms:** No errors, but overlay doesn't appear
**Debugging steps:**

1. **Check Sway outputs:**
```bash
swaymsg -t get_outputs
```

2. **Test with a simple colored overlay:**
```bash
# Try with different anchor positions
./target/release/face-overlay --anchor lower-left -vv
./target/release/face-overlay --anchor lower-right -vv
```

3. **Verify layer shell is working with other tools:**
```bash
# If you have waybar, wlr-randr, or similar tools, they should work
waybar --version  # These tools also use layer shell
```

### Issue 4: Sway Permission Issues
**Solution:**
```bash
# Make sure your user is in the correct groups
groups | grep -E "(video|input|wayland)"

# Check Sway configuration doesn't block overlays
grep -i "layer\|overlay" ~/.config/sway/config
```

## Testing Layer Shell Support

### Method 1: Use a Known Working Tool
```bash
# Install and test waybar (uses same layer shell protocol)
waybar &
# If waybar appears, layer shell works

# Or test with wlr-randr
wlr-randr --help
```

### Method 2: Minimal Layer Shell Test
Create a test to verify layer shell works:
```bash
# Run face-overlay with full debugging
RUST_LOG=debug ./target/release/face-overlay -vv 2>&1 | tee debug.log

# Check the debug.log for:
# - "zwlr_layer_shell_v1" in protocol list
# - "Layer surface created successfully"
# - No error messages about surface configuration
```

## Known Working Configurations

### ✅ Confirmed Working:
- **Sway 1.5+** with wlroots 0.12+
- **river** (wlroots-based)
- **dwl** (dwm for Wayland)
- **Hyprland** (supports layer shell)

### ❌ Won't Work:
- **GNOME on Wayland** (doesn't support wlr-layer-shell)
- **KDE Plasma Wayland** (different protocol)
- **Weston** (minimal compositor, limited layer shell)

## Advanced Debugging

### Enable Wayland Protocol Tracing
```bash
# Set environment variable for protocol debugging
export WAYLAND_DEBUG=1
./target/release/face-overlay -v 2>&1 | head -50
```

### Check Sway IPC
```bash
# Monitor Sway events while running face-overlay
swaymsg -t subscribe -m '["window"]' &
./target/release/face-overlay -v
```

### Verify Layer Shell Protocol Version
```bash
# Check exact protocol version
wayland-scanner client-header \
  /usr/share/wayland-protocols/unstable/wlr-layer-shell-unstable-v1/wlr-layer-shell-unstable-v1.xml \
  | grep -A5 "version"
```

## Workarounds for Common Problems

### 1. If Surface Appears But Is Invisible
Try forcing a background color by modifying the surface creation to set a solid color initially.

### 2. If Surface Is Too Small/Large
```bash
# Try different sizes
./target/release/face-overlay --width 200 --height 150 -v
./target/release/face-overlay --width 800 --height 600 -v
```

### 3. If Anchoring Doesn't Work
```bash
# Test all anchor positions
for anchor in lower-left lower-right; do
  echo "Testing anchor: $anchor"
  timeout 5s ./target/release/face-overlay --anchor $anchor -v
done
```

## Getting Help

If none of these solutions work, please provide:

1. **Environment info:**
```bash
echo "Compositor: $(echo $XDG_CURRENT_DESKTOP)"
echo "Session type: $(echo $XDG_SESSION_TYPE)"
swaymsg -t get_version
```

2. **Protocol availability:**
```bash
wayland-info | grep -E "(layer_shell|compositor|shm)"
```

3. **Debug output:**
```bash
RUST_LOG=debug ./target/release/face-overlay -vv --debug-wayland 2>&1 | head -30
```

This information will help identify exactly what's preventing the overlay from appearing in your Sway setup.