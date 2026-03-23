//! Mutual HMAC-SHA256 challenge-response authentication.
//!
//! Runs over raw TCP **before** any Cake protocol framing, so unauthenticated
//! peers never see valid Cake messages.
//!
//! Protocol (4 steps, 2 round trips):
//!   1. Master → Worker : 32-byte random nonce
//!   2. Worker → Master : HMAC-SHA256(key, master_nonce) ‖ 32-byte worker nonce
//!   3. Master verifies worker HMAC, then sends HMAC-SHA256(key, worker_nonce)
//!   4. Worker verifies master HMAC
//!
//! Both sides must possess the same pre-shared key.

use anyhow::Result;
use hmac::{Hmac, Mac};
use rand::RngCore;
use sha2::Sha256;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

type HmacSha256 = Hmac<Sha256>;

const NONCE_SIZE: usize = 32;
const HMAC_SIZE: usize = 32;

/// Compute HMAC-SHA256(key, data).
pub fn compute_hmac(key: &[u8], data: &[u8]) -> [u8; HMAC_SIZE] {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(data);
    mac.finalize().into_bytes().into()
}

/// Generate a random 32-byte nonce.
fn random_nonce() -> [u8; NONCE_SIZE] {
    let mut nonce = [0u8; NONCE_SIZE];
    rand::thread_rng().fill_bytes(&mut nonce);
    nonce
}

/// Constant-time comparison of two byte slices.
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}

/// Master side of the mutual authentication.
///
/// Called on the TCP stream immediately after connection, before any Cake
/// protocol messages.
pub async fn authenticate_as_master<S>(stream: &mut S, key: &str) -> Result<()>
where
    S: AsyncReadExt + AsyncWriteExt + Unpin,
{
    let key_bytes = key.as_bytes();

    // Step 1: send nonce to worker
    let master_nonce = random_nonce();
    stream.write_all(&master_nonce).await?;
    stream.flush().await?;

    // Step 2: read worker's HMAC response + worker's nonce in one call
    let mut response = [0u8; HMAC_SIZE + NONCE_SIZE];
    stream.read_exact(&mut response).await?;
    let worker_hmac = &response[..HMAC_SIZE];
    let worker_nonce = &response[HMAC_SIZE..];

    // Step 3: verify worker's HMAC
    let expected = compute_hmac(key_bytes, &master_nonce);
    if !constant_time_eq(worker_hmac, &expected) {
        return Err(anyhow!("worker authentication failed: invalid HMAC"));
    }

    // Step 4: send master's HMAC response
    let master_hmac = compute_hmac(key_bytes, worker_nonce);
    stream.write_all(&master_hmac).await?;
    stream.flush().await?;

    Ok(())
}

/// Worker side of the mutual authentication.
///
/// Called on the accepted TCP stream before reading any Cake protocol messages.
pub async fn authenticate_as_worker<S>(stream: &mut S, key: &str) -> Result<()>
where
    S: AsyncReadExt + AsyncWriteExt + Unpin,
{
    let key_bytes = key.as_bytes();

    // Step 1: read master's nonce
    let mut master_nonce = [0u8; NONCE_SIZE];
    stream.read_exact(&mut master_nonce).await?;

    // Step 2: send HMAC response + our nonce in one write
    let worker_hmac = compute_hmac(key_bytes, &master_nonce);
    let worker_nonce = random_nonce();
    let mut response = [0u8; HMAC_SIZE + NONCE_SIZE];
    response[..HMAC_SIZE].copy_from_slice(&worker_hmac);
    response[HMAC_SIZE..].copy_from_slice(&worker_nonce);
    stream.write_all(&response).await?;
    stream.flush().await?;

    // Step 3: read master's HMAC response
    let mut master_hmac = [0u8; HMAC_SIZE];
    stream.read_exact(&mut master_hmac).await?;

    // Step 4: verify master's HMAC
    let expected = compute_hmac(key_bytes, &worker_nonce);
    if !constant_time_eq(&master_hmac, &expected) {
        return Err(anyhow!("master authentication failed: invalid HMAC"));
    }

    Ok(())
}
