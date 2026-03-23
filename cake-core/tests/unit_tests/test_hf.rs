//! Tests for HuggingFace utility functions.

use cake_core::utils::hf;

#[test]
fn test_looks_like_hf_repo_valid() {
    assert!(hf::looks_like_hf_repo("org/model"));
    assert!(hf::looks_like_hf_repo("evilsocket/Qwen3-0.6B"));
}

#[test]
fn test_looks_like_hf_repo_invalid() {
    assert!(!hf::looks_like_hf_repo(""));
    assert!(!hf::looks_like_hf_repo("model"));
    assert!(!hf::looks_like_hf_repo("/model"));
    assert!(!hf::looks_like_hf_repo("./model"));
    assert!(!hf::looks_like_hf_repo("~/model"));
    assert!(!hf::looks_like_hf_repo("org/"));
    assert!(!hf::looks_like_hf_repo("/org/model"));
    assert!(!hf::looks_like_hf_repo("a/b/c"));
}
