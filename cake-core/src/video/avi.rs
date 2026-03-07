//! Pure-Rust uncompressed AVI writer (RIFF/AVI 1.0).
//!
//! Writes an AVI file containing a single video stream with uncompressed
//! RGB24 (DIB) frames. The resulting file is playable by VLC, ffmpeg,
//! QuickTime, Windows Media Player, and most other video software.
//!
//! AVI 1.0 with uncompressed frames has a theoretical 2 GB RIFF size limit.
//! For the frame counts and resolutions used in LTX-Video generation this
//! is more than sufficient (41 frames @ 512x704 ≈ 44 MB).

use image::{ImageBuffer, Rgb};
use std::io::Write;

/// Write an uncompressed AVI to any `Write` sink.
///
/// Frames must all be the same dimensions. Each frame is stored as a
/// bottom-up DIB (the AVI/BMP convention), with row order flipped.
pub fn write_avi<W: Write>(
    w: &mut W,
    frames: &[ImageBuffer<Rgb<u8>, Vec<u8>>],
    fps: usize,
    width: u32,
    height: u32,
) -> anyhow::Result<()> {
    if frames.is_empty() {
        anyhow::bail!("cannot write AVI with zero frames");
    }
    if fps == 0 {
        anyhow::bail!("fps must be > 0");
    }

    let num_frames = frames.len() as u32;
    // Each row is padded to 4-byte boundary (RGB24 = 3 bytes per pixel)
    let row_bytes = width * 3;
    let row_stride = (row_bytes + 3) & !3; // pad to 4-byte boundary
    let frame_size = row_stride * height; // raw DIB frame size
    let usec_per_frame = 1_000_000u32 / fps as u32;

    // movi list: each frame is a "00dc" chunk (4 byte tag + 4 byte size + data)
    let movi_payload_size: u32 = num_frames * (8 + frame_size);
    let movi_list_size: u32 = 4 + movi_payload_size; // "movi" + chunks

    // hdrl list size
    let avih_chunk_size: u32 = 8 + 56; // "avih" + size_u32 + 56 bytes payload
    let strh_chunk_size: u32 = 8 + 56; // "strh" + size_u32 + 56 bytes payload
    let strf_chunk_size: u32 = 8 + 40; // "strf" + size_u32 + BITMAPINFOHEADER(40)
    let strl_list_size: u32 = 4 + strh_chunk_size + strf_chunk_size; // "strl" + chunks
    let hdrl_list_size: u32 = 4 + avih_chunk_size + 8 + strl_list_size; // "hdrl" + avih + LIST(strl)

    // idx1 chunk: 8 byte header + 16 bytes per frame
    let idx1_chunk_size: u32 = 8 + num_frames * 16;

    // Total RIFF size: "AVI " + LIST(hdrl) + LIST(movi) + idx1
    let riff_size: u32 = 4 + (8 + hdrl_list_size) + (8 + movi_list_size) + idx1_chunk_size;

    // ── RIFF header ──────────────────────────────────────────────
    w.write_all(b"RIFF")?;
    w.write_all(&riff_size.to_le_bytes())?;
    w.write_all(b"AVI ")?;

    // ── hdrl LIST ────────────────────────────────────────────────
    w.write_all(b"LIST")?;
    w.write_all(&hdrl_list_size.to_le_bytes())?;
    w.write_all(b"hdrl")?;

    // ── avih (main AVI header) ───────────────────────────────────
    w.write_all(b"avih")?;
    w.write_all(&56u32.to_le_bytes())?; // size of avih data
    w.write_all(&usec_per_frame.to_le_bytes())?; // dwMicroSecPerFrame
    w.write_all(&(frame_size * fps as u32).to_le_bytes())?; // dwMaxBytesPerSec
    w.write_all(&0u32.to_le_bytes())?; // dwPaddingGranularity
    w.write_all(&0x10u32.to_le_bytes())?; // dwFlags: AVIF_HASINDEX (0x10)
    w.write_all(&num_frames.to_le_bytes())?; // dwTotalFrames
    w.write_all(&0u32.to_le_bytes())?; // dwInitialFrames
    w.write_all(&1u32.to_le_bytes())?; // dwStreams
    w.write_all(&frame_size.to_le_bytes())?; // dwSuggestedBufferSize
    w.write_all(&width.to_le_bytes())?; // dwWidth
    w.write_all(&height.to_le_bytes())?; // dwHeight
    w.write_all(&[0u8; 16])?; // dwReserved[4]

    // ── strl LIST (stream list) ──────────────────────────────────
    w.write_all(b"LIST")?;
    w.write_all(&strl_list_size.to_le_bytes())?;
    w.write_all(b"strl")?;

    // ── strh (stream header) ─────────────────────────────────────
    w.write_all(b"strh")?;
    w.write_all(&56u32.to_le_bytes())?; // size of strh data
    w.write_all(b"vids")?; // fccType: video stream
    w.write_all(&0u32.to_le_bytes())?; // fccHandler: 0 = uncompressed DIB
    w.write_all(&0u32.to_le_bytes())?; // dwFlags
    w.write_all(&0u16.to_le_bytes())?; // wPriority
    w.write_all(&0u16.to_le_bytes())?; // wLanguage
    w.write_all(&0u32.to_le_bytes())?; // dwInitialFrames
    w.write_all(&1u32.to_le_bytes())?; // dwScale
    w.write_all(&(fps as u32).to_le_bytes())?; // dwRate
    w.write_all(&0u32.to_le_bytes())?; // dwStart
    w.write_all(&num_frames.to_le_bytes())?; // dwLength
    w.write_all(&frame_size.to_le_bytes())?; // dwSuggestedBufferSize
    w.write_all(&0xFFFFFFFFu32.to_le_bytes())?; // dwQuality (-1 = default)
    w.write_all(&0u32.to_le_bytes())?; // dwSampleSize
    w.write_all(&0u16.to_le_bytes())?; // rcFrame.left
    w.write_all(&0u16.to_le_bytes())?; // rcFrame.top
    w.write_all(&(width as u16).to_le_bytes())?; // rcFrame.right
    w.write_all(&(height as u16).to_le_bytes())?; // rcFrame.bottom

    // ── strf (stream format = BITMAPINFOHEADER) ──────────────────
    w.write_all(b"strf")?;
    w.write_all(&40u32.to_le_bytes())?; // size of BITMAPINFOHEADER
    w.write_all(&40u32.to_le_bytes())?; // biSize
    w.write_all(&width.to_le_bytes())?; // biWidth
    w.write_all(&height.to_le_bytes())?; // biHeight (positive = bottom-up)
    w.write_all(&1u16.to_le_bytes())?; // biPlanes
    w.write_all(&24u16.to_le_bytes())?; // biBitCount (RGB24)
    w.write_all(&0u32.to_le_bytes())?; // biCompression (BI_RGB = 0)
    w.write_all(&frame_size.to_le_bytes())?; // biSizeImage
    w.write_all(&0u32.to_le_bytes())?; // biXPelsPerMeter
    w.write_all(&0u32.to_le_bytes())?; // biYPelsPerMeter
    w.write_all(&0u32.to_le_bytes())?; // biClrUsed
    w.write_all(&0u32.to_le_bytes())?; // biClrImportant

    // ── movi LIST ────────────────────────────────────────────────
    w.write_all(b"LIST")?;
    w.write_all(&movi_list_size.to_le_bytes())?;
    w.write_all(b"movi")?;

    // Row buffer for bottom-up flip + RGB→BGR + row padding
    let mut row_buf = vec![0u8; row_stride as usize];

    for frame in frames {
        w.write_all(b"00dc")?; // chunk ID: stream 0, compressed (dc)
        w.write_all(&frame_size.to_le_bytes())?;

        // AVI DIB frames are bottom-up: write rows in reverse order
        // Also convert RGB to BGR (BMP/AVI convention)
        for y in (0..height).rev() {
            let row_start = (y * width * 3) as usize;
            let row_end = row_start + (width * 3) as usize;
            let src = &frame.as_raw()[row_start..row_end];

            // Convert RGB -> BGR
            for x in 0..width as usize {
                row_buf[x * 3] = src[x * 3 + 2]; // B
                row_buf[x * 3 + 1] = src[x * 3 + 1]; // G
                row_buf[x * 3 + 2] = src[x * 3]; // R
            }
            // Padding bytes are already zeroed from vec initialization
            w.write_all(&row_buf)?;
        }
    }

    // ── idx1 (AVI 1.0 index) ─────────────────────────────────────
    let idx1_size = num_frames * 16; // 16 bytes per entry
    w.write_all(b"idx1")?;
    w.write_all(&idx1_size.to_le_bytes())?;

    let mut offset: u32 = 4; // offset from start of movi data (after "movi" tag)
    for _ in 0..num_frames {
        w.write_all(b"00dc")?; // ckid
        w.write_all(&0x10u32.to_le_bytes())?; // dwFlags: AVIIF_KEYFRAME
        w.write_all(&offset.to_le_bytes())?; // dwOffset
        w.write_all(&frame_size.to_le_bytes())?; // dwSize
        offset += 8 + frame_size; // skip chunk header (tag + size) + data
    }

    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_frame(width: u32, height: u32, color: [u8; 3]) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        ImageBuffer::from_fn(width, height, |_, _| Rgb(color))
    }

    #[test]
    fn test_write_avi_basic() {
        let frames = vec![
            make_test_frame(8, 6, [255, 0, 0]),
            make_test_frame(8, 6, [0, 255, 0]),
            make_test_frame(8, 6, [0, 0, 255]),
        ];
        let mut buf = Vec::new();
        write_avi(&mut buf, &frames, 24, 8, 6).unwrap();

        // Check RIFF header
        assert_eq!(&buf[0..4], b"RIFF");
        assert_eq!(&buf[8..12], b"AVI ");

        // Verify total size matches RIFF size field + 8
        let riff_size = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        assert_eq!(buf.len() as u32, riff_size + 8);
    }

    #[test]
    fn test_write_avi_empty_fails() {
        let frames: Vec<ImageBuffer<Rgb<u8>, Vec<u8>>> = vec![];
        let mut buf = Vec::new();
        assert!(write_avi(&mut buf, &frames, 24, 8, 6).is_err());
    }

    #[test]
    fn test_write_avi_zero_fps_fails() {
        let frames = vec![make_test_frame(8, 6, [0, 0, 0])];
        let mut buf = Vec::new();
        assert!(write_avi(&mut buf, &frames, 0, 8, 6).is_err());
    }

    #[test]
    fn test_write_avi_single_frame() {
        let frames = vec![make_test_frame(4, 4, [128, 64, 32])];
        let mut buf = Vec::new();
        write_avi(&mut buf, &frames, 1, 4, 4).unwrap();

        let riff_size = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        assert_eq!(buf.len() as u32, riff_size + 8);
    }

    #[test]
    fn test_write_avi_odd_width_padding() {
        // Width=5, RGB24: 5*3=15 bytes/row, padded to 16 (next multiple of 4)
        let frames = vec![make_test_frame(5, 3, [255, 128, 0])];
        let mut buf = Vec::new();
        write_avi(&mut buf, &frames, 30, 5, 3).unwrap();

        let riff_size = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        assert_eq!(buf.len() as u32, riff_size + 8);
    }

    #[test]
    fn test_video_output_roundtrip() {
        use crate::video::VideoOutput;

        let frames = vec![
            make_test_frame(8, 6, [255, 0, 0]),
            make_test_frame(8, 6, [0, 255, 0]),
        ];
        let output = VideoOutput::new(frames, 24, 8, 6);
        assert_eq!(output.num_frames(), 2);
        assert!((output.duration_secs() - 2.0 / 24.0).abs() < 0.001);

        let avi_bytes = output.to_avi().unwrap();
        assert_eq!(&avi_bytes[0..4], b"RIFF");
    }
}
