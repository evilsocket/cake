//! Video output types and pure-Rust AVI muxer.
//!
//! No third-party codec dependencies — writes uncompressed RGB24 AVI
//! that any video player, ffmpeg, or browser can read. Users can
//! transcode to H.264/H.265 externally if compression is needed.

mod avi;

use image::{ImageBuffer, Rgb};

pub use avi::write_avi;

/// Complete video output from a generation pipeline.
pub struct VideoOutput {
    /// Individual frames in RGB8 format, ordered chronologically.
    pub frames: Vec<ImageBuffer<Rgb<u8>, Vec<u8>>>,
    /// Frames per second.
    pub fps: usize,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
}

impl VideoOutput {
    /// Create a VideoOutput from frames and metadata.
    pub fn new(
        frames: Vec<ImageBuffer<Rgb<u8>, Vec<u8>>>,
        fps: usize,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            frames,
            fps,
            width,
            height,
        }
    }

    /// Encode this video as an uncompressed AVI file in memory.
    pub fn to_avi(&self) -> anyhow::Result<Vec<u8>> {
        let mut buf = Vec::new();
        write_avi(&mut buf, &self.frames, self.fps, self.width, self.height)?;
        Ok(buf)
    }

    /// Write this video as an AVI file to the given path.
    pub fn save_avi(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let mut file = std::fs::File::create(path)?;
        write_avi(
            &mut file,
            &self.frames,
            self.fps,
            self.width,
            self.height,
        )
    }

    /// Save individual frames as numbered PNG files in the given directory.
    pub fn save_frames(&self, dir: &std::path::Path, prefix: &str) -> anyhow::Result<()> {
        std::fs::create_dir_all(dir)?;
        for (i, frame) in self.frames.iter().enumerate() {
            let path = dir.join(format!("{}_{:04}.png", prefix, i));
            frame.save(&path)?;
        }
        Ok(())
    }

    /// Total number of frames.
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    /// Duration in seconds.
    pub fn duration_secs(&self) -> f64 {
        if self.fps == 0 {
            return 0.0;
        }
        self.frames.len() as f64 / self.fps as f64
    }
}
