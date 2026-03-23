//! Tests for model listing utilities.

use cake_core::utils::models::{find_model, list_models};

#[test]
fn test_list_models_returns_vec() {
    // Should not panic even if no models cached
    let models = list_models().unwrap();
    // Can be empty, just verify it returns
    // Just verify it returns without panicking; may be empty if no models cached
    let _ = models.len();
}

#[test]
fn test_find_model_nonexistent() {
    let result = find_model("nonexistent/model-xyz-999").unwrap();
    assert!(result.is_none());
}
