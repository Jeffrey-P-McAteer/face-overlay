#!/bin/bash

# Sway Overlay Test Script
# Run this when you're in a Sway session to test overlay functionality

echo "🔍 Face Overlay - Sway Compatibility Test"
echo "========================================"

# Check if we're in Wayland
echo "1. Checking Wayland environment..."
if [ -z "$WAYLAND_DISPLAY" ]; then
    echo "❌ WAYLAND_DISPLAY not set - not in Wayland session"
    echo "   Please log into Sway or another Wayland compositor"
    exit 1
else
    echo "✅ WAYLAND_DISPLAY = $WAYLAND_DISPLAY"
fi

if [ "$XDG_SESSION_TYPE" != "wayland" ]; then
    echo "⚠️  XDG_SESSION_TYPE = $XDG_SESSION_TYPE (expected 'wayland')"
else
    echo "✅ XDG_SESSION_TYPE = $XDG_SESSION_TYPE"
fi

# Check if Sway is running
echo
echo "2. Checking Sway..."
if command -v swaymsg >/dev/null 2>&1; then
    if swaymsg -t get_version >/dev/null 2>&1; then
        SWAY_VERSION=$(swaymsg -t get_version | grep -o 'swayVersion":"[^"]*' | cut -d'"' -f3)
        echo "✅ Sway running, version: $SWAY_VERSION"
    else
        echo "⚠️  swaymsg found but Sway not responding"
        echo "   You might be in a different Wayland compositor"
    fi
else
    echo "⚠️  swaymsg not found"
    echo "   You might be using a different Wayland compositor"
fi

# Check if face-overlay binary exists
echo
echo "3. Checking face-overlay binary..."
if [ -f "./target/release/face-overlay" ]; then
    echo "✅ face-overlay binary found"
else
    echo "❌ face-overlay binary not found"
    echo "   Run: cargo build --release"
    exit 1
fi

# Test Wayland protocols
echo
echo "4. Testing Wayland connection and protocols..."
echo "Running face-overlay with full debugging for 5 seconds..."
echo "Look for 'zwlr_layer_shell_v1' in the protocol list below:"
echo

timeout 5s ./target/release/face-overlay -vv --debug-wayland 2>&1 | head -20

echo
echo "5. Test Results"
echo "==============="
echo "If you saw:"
echo "✅ 'zwlr_layer_shell_v1' in protocol list → Layer shell supported"
echo "✅ 'Layer surface created successfully' → Overlay should work"
echo "✅ 'Surface should appear in [corner] corner' → Check that corner!"
echo
echo "If you saw errors:"
echo "❌ 'Layer shell not available' → Compositor doesn't support overlays"
echo "❌ 'Could not find wayland compositor' → Not in Wayland session"
echo
echo "6. Manual Test"
echo "=============="
echo "To test manually with different positions:"
echo "  ./target/release/face-overlay --anchor lower-left -v"
echo "  ./target/release/face-overlay --anchor lower-right -v"
echo
echo "The overlay should appear as a colored rectangle in the specified corner."
echo "For more help, see: SWAY_DEBUGGING.md"