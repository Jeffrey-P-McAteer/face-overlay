use std::time::Instant;
use tracing::{debug, warn};

pub struct MouseTracker {
    last_position: (i32, i32),
    last_check: Instant,
    check_interval_ms: u64,
}

impl MouseTracker {
    pub fn new() -> Self {
        Self {
            last_position: (0, 0),
            last_check: Instant::now(),
            check_interval_ms: 16, // ~60 FPS checking
        }
    }

    /// Get current mouse position using multiple fallback methods
    pub fn get_mouse_position(&mut self) -> Option<(i32, i32)> {
        // Throttle position checks to avoid excessive system calls
        if self.last_check.elapsed().as_millis() < self.check_interval_ms as u128 {
            return Some(self.last_position);
        }
        
        self.last_check = Instant::now();
        
        // Try X11 method first (works even in Wayland with XWayland)
        if let Some(pos) = self.get_mouse_position_x11() {
            self.last_position = pos;
            return Some(pos);
        }
        
        // Fallback to /proc/bus/input/devices parsing
        if let Some(pos) = self.get_mouse_position_proc() {
            self.last_position = pos;
            return Some(pos);
        }
        
        // Return last known position if all methods fail
        Some(self.last_position)
    }
    
    /// Get mouse position using X11 (works with XWayland)
    fn get_mouse_position_x11(&self) -> Option<(i32, i32)> {
        use std::process::Command;
        
        // Try xdotool first
        if let Ok(output) = Command::new("xdotool")
            .args(&["getmouselocation", "--shell"])
            .output() 
        {
            if output.status.success() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                let mut x = None;
                let mut y = None;
                
                for line in output_str.lines() {
                    if line.starts_with("X=") {
                        x = line[2..].parse().ok();
                    } else if line.starts_with("Y=") {
                        y = line[2..].parse().ok();
                    }
                }
                
                if let (Some(x), Some(y)) = (x, y) {
                    debug!("Mouse position from xdotool: ({}, {})", x, y);
                    return Some((x, y));
                }
            }
        }
        
        // Fallback to xwininfo + xdpyinfo
        if let Ok(output) = Command::new("sh")
            .arg("-c")
            .arg("eval $(xdotool getmouselocation --shell 2>/dev/null || echo 'X=0'; echo 'Y=0'); echo $X,$Y")
            .output()
        {
            if output.status.success() {
                let coords = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if let Some((x_str, y_str)) = coords.split_once(',') {
                    if let (Ok(x), Ok(y)) = (x_str.parse::<i32>(), y_str.parse::<i32>()) {
                        debug!("Mouse position from shell: ({}, {})", x, y);
                        return Some((x, y));
                    }
                }
            }
        }
        
        None
    }
    
    /// Get mouse position from /proc filesystem (Linux-specific)
    fn get_mouse_position_proc(&self) -> Option<(i32, i32)> {
        // This is a simplified approach - in practice, parsing /proc/bus/input/devices
        // and reading from the appropriate event device would be more reliable
        warn!("Proc-based mouse tracking not fully implemented");
        None
    }
    
    /// Check if mouse is in the specified rectangle
    pub fn is_mouse_in_area(&mut self, x: i32, y: i32, width: i32, height: i32) -> bool {
        if let Some((mouse_x, mouse_y)) = self.get_mouse_position() {
            mouse_x >= x && mouse_x < x + width && mouse_y >= y && mouse_y < y + height
        } else {
            false
        }
    }
    
    /// Check if mouse is near the specified rectangle (with margin)
    pub fn is_mouse_near_area(&mut self, x: i32, y: i32, width: i32, height: i32, margin: i32) -> bool {
        if let Some((mouse_x, mouse_y)) = self.get_mouse_position() {
            mouse_x >= x - margin && 
            mouse_x < x + width + margin && 
            mouse_y >= y - margin && 
            mouse_y < y + height + margin
        } else {
            false
        }
    }
}