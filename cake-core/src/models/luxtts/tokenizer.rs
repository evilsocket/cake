//! LuxTTS tokenizer — maps text to phoneme token IDs.
//!
//! Uses a CMUdict-IPA dictionary for word→IPA lookup, with rule-based fallback
//! for out-of-vocabulary words. Then maps IPA symbols to token IDs via tokens.txt.

use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

/// Phonemizer: text → IPA phoneme sequence → token IDs.
pub struct Phonemizer {
    /// Word → IPA transcription (lowercase keys).
    ipa_dict: HashMap<String, String>,
    /// IPA symbol → token ID.
    token_map: HashMap<String, u32>,
    /// Token ID for space/word boundary.
    space_id: Option<u32>,
    /// Token ID for silence/padding.
    #[allow(dead_code)]
    pad_id: u32,
}

impl Phonemizer {
    /// Load from tokens.txt (TSV: token\tID) and optional CMUdict-IPA file.
    pub fn load(tokens_path: &Path, dict_path: Option<&Path>) -> Result<Self> {
        let token_map = Self::load_tokens(tokens_path)?;

        let ipa_dict = if let Some(dp) = dict_path {
            Self::load_cmudict_ipa(dp)?
        } else {
            HashMap::new()
        };

        let space_id = token_map.get(" ").or_else(|| token_map.get("sp")).copied();
        let pad_id = token_map.get("<pad>").or_else(|| token_map.get("<blank>")).copied().unwrap_or(0);

        Ok(Self {
            ipa_dict,
            token_map,
            space_id,
            pad_id,
        })
    }

    /// Parse tokens.txt (each line: `token\tindex`).
    /// Uses the LAST tab as separator so that tokens containing spaces work correctly.
    fn load_tokens(path: &Path) -> Result<HashMap<String, u32>> {
        let content = std::fs::read_to_string(path)?;
        let mut map = HashMap::new();
        for line in content.lines() {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            // Split on last tab to handle tokens that might be spaces
            if let Some(tab_pos) = line.rfind('\t') {
                let token = &line[..tab_pos];
                let id_str = line[tab_pos + 1..].trim();
                if let Ok(id) = id_str.parse::<u32>() {
                    map.insert(token.to_string(), id);
                }
            }
        }
        Ok(map)
    }

    /// Load CMUdict-IPA dictionary (format: `WORD\tIPA` or `WORD  IPA`).
    fn load_cmudict_ipa(path: &Path) -> Result<HashMap<String, String>> {
        let content = std::fs::read_to_string(path)?;
        let mut dict = HashMap::new();
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with(";;;") {
                continue;
            }
            // Try tab separator first, then double space
            let parts: Vec<&str> = if line.contains('\t') {
                line.splitn(2, '\t').collect()
            } else {
                line.splitn(2, "  ").collect()
            };
            if parts.len() == 2 {
                let word = parts[0].trim().to_lowercase();
                // Take the first pronunciation variant (before comma if multiple)
                let ipa = parts[1].trim().split(',').next().unwrap_or("").trim();
                if !ipa.is_empty() {
                    dict.insert(word, ipa.to_string());
                }
            }
        }
        Ok(dict)
    }

    /// Convert text to a sequence of token IDs.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let text = text.to_lowercase();
        let mut tokens = Vec::new();

        // Split into words and punctuation
        let words: Vec<&str> = text.split_whitespace().collect();

        for (i, word) in words.iter().enumerate() {
            if i > 0 {
                if let Some(sp) = self.space_id {
                    tokens.push(sp);
                }
            }

            // Strip punctuation from edges
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
            if clean.is_empty() {
                // Pure punctuation — look up directly
                let punct_tokens = self.map_ipa_to_tokens(word);
                tokens.extend(punct_tokens);
                continue;
            }

            // Look up in dictionary
            if let Some(ipa) = self.ipa_dict.get(clean) {
                let ipa_tokens = self.map_ipa_to_tokens(ipa);
                tokens.extend(ipa_tokens);
            } else {
                // OOV fallback: rule-based letter-to-IPA
                let ipa = self.rules_fallback(clean);
                let ipa_tokens = self.map_ipa_to_tokens(&ipa);
                tokens.extend(ipa_tokens);
            }
        }

        Ok(tokens)
    }

    /// Map an IPA string to token IDs by greedily matching the longest token.
    fn map_ipa_to_tokens(&self, ipa: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = ipa.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let mut best_len = 0;
            let mut best_id = None;

            // Try matching longest substring first (up to 3 chars for digraphs/trigraphs)
            for len in (1..=3.min(chars.len() - i)).rev() {
                let substr: String = chars[i..i + len].iter().collect();
                if let Some(&id) = self.token_map.get(&substr) {
                    best_len = len;
                    best_id = Some(id);
                    break;
                }
            }

            if let Some(id) = best_id {
                tokens.push(id);
                i += best_len;
            } else {
                // Unknown character — skip
                i += 1;
            }
        }

        tokens
    }

    /// Simple rule-based English grapheme→IPA fallback for OOV words.
    fn rules_fallback(&self, word: &str) -> String {
        let mut ipa = String::new();
        let chars: Vec<char> = word.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let remaining = &word[i..];
            // Try digraphs first
            let (phoneme, advance) = if remaining.len() >= 2 {
                match &remaining[..2] {
                    "th" => ("\u{03B8}", 2), // θ
                    "sh" => ("\u{0283}", 2), // ʃ
                    "ch" => ("t\u{0283}", 2), // tʃ
                    "ng" => ("\u{014B}", 2), // ŋ
                    "ph" => ("f", 2),
                    "wh" => ("w", 2),
                    "ck" => ("k", 2),
                    "ee" => ("i\u{02D0}", 2), // iː
                    "oo" => ("u\u{02D0}", 2), // uː
                    "ou" => ("a\u{028A}", 2), // aʊ
                    "ow" => ("a\u{028A}", 2), // aʊ
                    "ai" => ("e\u{026A}", 2), // eɪ
                    "ay" => ("e\u{026A}", 2), // eɪ
                    "ea" => ("i\u{02D0}", 2), // iː
                    _ => ("", 0),
                }
            } else {
                ("", 0)
            };

            if advance > 0 {
                ipa.push_str(phoneme);
                i += advance;
                continue;
            }

            // Single character mapping
            let phoneme = match chars[i] {
                'a' => "\u{00E6}", // æ
                'b' => "b",
                'c' => "k",
                'd' => "d",
                'e' => "\u{025B}", // ɛ
                'f' => "f",
                'g' => "g",
                'h' => "h",
                'i' => "\u{026A}", // ɪ
                'j' => "d\u{0292}", // dʒ
                'k' => "k",
                'l' => "l",
                'm' => "m",
                'n' => "n",
                'o' => "\u{0251}", // ɑ
                'p' => "p",
                'q' => "k",
                'r' => "\u{0279}", // ɹ
                's' => "s",
                't' => "t",
                'u' => "\u{028C}", // ʌ
                'v' => "v",
                'w' => "w",
                'x' => "ks",
                'y' => "j",
                'z' => "z",
                _ => "",
            };
            ipa.push_str(phoneme);
            i += 1;
        }

        ipa
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_test_tokens() -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, "<pad>\t0").unwrap();
        writeln!(f, "h\t1").unwrap();
        writeln!(f, "\u{025B}\t2").unwrap(); // ɛ
        writeln!(f, "l\t3").unwrap();
        writeln!(f, "o\u{028A}\t4").unwrap(); // oʊ
        writeln!(f, " \t5").unwrap(); // space
        writeln!(f, "w\t6").unwrap();
        writeln!(f, "\u{025D}\t7").unwrap(); // ɝ
        writeln!(f, "d\t8").unwrap();
        f
    }

    fn make_test_dict() -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, "hello\th\u{025B}l\u{02C8}o\u{028A}").unwrap();
        writeln!(f, "world\tw\u{025D}ld").unwrap();
        f
    }

    #[test]
    fn test_load_tokens() {
        let tokens_file = make_test_tokens();
        let map = Phonemizer::load_tokens(tokens_file.path()).unwrap();
        assert_eq!(map.get("<pad>"), Some(&0));
        assert_eq!(map.get("h"), Some(&1));
        assert_eq!(map.get(" "), Some(&5));
    }

    #[test]
    fn test_load_phonemizer() {
        let tokens_file = make_test_tokens();
        let dict_file = make_test_dict();
        let p = Phonemizer::load(tokens_file.path(), Some(dict_file.path())).unwrap();
        assert!(p.ipa_dict.contains_key("hello"));
        assert!(p.ipa_dict.contains_key("world"));
    }

    #[test]
    fn test_tokenize_known_word() {
        let tokens_file = make_test_tokens();
        let dict_file = make_test_dict();
        let p = Phonemizer::load(tokens_file.path(), Some(dict_file.path())).unwrap();
        let ids = p.tokenize("hello").unwrap();
        // Should produce tokens for h, ɛ, l, oʊ (skipping ˈ stress marker)
        assert!(!ids.is_empty());
        // First token should be 'h' = 1
        assert_eq!(ids[0], 1);
    }

    #[test]
    fn test_rules_fallback() {
        let tokens_file = make_test_tokens();
        let p = Phonemizer::load(tokens_file.path(), None).unwrap();
        let ipa = p.rules_fallback("cat");
        // c→k, a→æ, t→t
        assert!(ipa.contains('k'));
        assert!(ipa.contains('\u{00E6}')); // æ
        assert!(ipa.contains('t'));
    }
}
