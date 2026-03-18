use constriction::stream::model::DefaultContiguousCategoricalEntropyModel;
use constriction::stream::stack::DefaultAnsCoder;
use constriction::stream::Decode;
use kvforge_core::codec::{EntropyCodec, EntropyConfig};
use kvforge_core::error::{KvForgeError, Result};
use kvforge_core::types::EntropyState;

/// CPU entropy codec using rANS via the constriction crate.
#[derive(Debug, Clone)]
pub struct CpuEntropyCodec;

impl CpuEntropyCodec {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuEntropyCodec {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a frequency histogram from byte data (256 bins).
fn compute_frequencies(data: &[u8]) -> Vec<u32> {
    let mut freq = vec![0u32; 256];
    for &b in data {
        freq[b as usize] += 1;
    }
    freq
}

/// Convert frequency counts to floating-point probabilities.
/// Ensures no symbol has zero probability (add-one smoothing).
fn frequencies_to_probabilities(freq: &[u32]) -> Vec<f64> {
    // Add-one smoothing to ensure all symbols have nonzero probability
    let smoothed: Vec<f64> = freq.iter().map(|&f| (f as f64) + 1.0).collect();
    let total: f64 = smoothed.iter().sum();
    smoothed.iter().map(|&f| f / total).collect()
}

impl EntropyCodec for CpuEntropyCodec {
    fn encode(&self, data: &[u8], config: &EntropyConfig) -> Result<(Vec<u8>, EntropyState)> {
        if !config.enabled || data.is_empty() {
            let state = EntropyState {
                frequencies: vec![],
                original_len: data.len(),
            };
            return Ok((data.to_vec(), state));
        }

        let frequencies = compute_frequencies(data);
        let probabilities = frequencies_to_probabilities(&frequencies);

        let model =
            DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast(
                &probabilities,
                None,
            )
            .map_err(|e| KvForgeError::EntropyError(format!("Failed to build model: {:?}", e)))?;

        let mut coder = DefaultAnsCoder::new();

        // ANS is stack-based: encode in reverse order so decode comes out in forward order
        let symbols: Vec<usize> = data.iter().map(|&b| b as usize).collect();
        coder
            .encode_iid_symbols_reverse(&symbols, &model)
            .map_err(|e| KvForgeError::EntropyError(format!("Encode failed: {}", e)))?;

        // Get compressed data as Vec<u32>, then convert to bytes
        let compressed_words = coder
            .into_compressed()
            .map_err(|e| KvForgeError::EntropyError(format!("Finalize failed: {:?}", e)))?;

        let mut compressed_bytes = Vec::with_capacity(compressed_words.len() * 4);
        for word in &compressed_words {
            compressed_bytes.extend_from_slice(&word.to_le_bytes());
        }

        let state = EntropyState {
            frequencies,
            original_len: data.len(),
        };

        Ok((compressed_bytes, state))
    }

    fn decode(&self, data: &[u8], state: &EntropyState) -> Result<Vec<u8>> {
        if state.frequencies.is_empty() || state.original_len == 0 {
            return Ok(data.to_vec());
        }

        // Convert bytes back to u32 words
        if !data.len().is_multiple_of(4) {
            return Err(KvForgeError::EntropyError(
                "Compressed data length not aligned to 4 bytes".into(),
            ));
        }
        let mut words = Vec::with_capacity(data.len() / 4);
        for chunk in data.chunks_exact(4) {
            words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }

        let probabilities = frequencies_to_probabilities(&state.frequencies);

        let model =
            DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast(
                &probabilities,
                None,
            )
            .map_err(|e| KvForgeError::EntropyError(format!("Failed to build model: {:?}", e)))?;

        let mut coder = DefaultAnsCoder::from_compressed(words)
            .map_err(|e| KvForgeError::EntropyError(format!("Failed to init decoder: {:?}", e)))?;

        let decoded: std::result::Result<Vec<usize>, _> = coder
            .decode_iid_symbols(state.original_len, &model)
            .collect();
        let decoded = decoded
            .map_err(|e| KvForgeError::EntropyError(format!("Decode failed: {}", e)))?;

        Ok(decoded.iter().map(|&s| s as u8).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_basic() {
        let codec = CpuEntropyCodec::new();
        let config = EntropyConfig { enabled: true };

        let data: Vec<u8> = (0..256).map(|i| (i % 256) as u8).collect();
        let (encoded, state) = codec.encode(&data, &config).unwrap();
        let decoded = codec.decode(&encoded, &state).unwrap();

        assert_eq!(data, decoded);
    }

    #[test]
    fn test_roundtrip_skewed() {
        let codec = CpuEntropyCodec::new();
        let config = EntropyConfig { enabled: true };

        // Highly skewed distribution — mostly zeros
        let mut data = vec![0u8; 1000];
        data[100] = 1;
        data[200] = 2;
        data[300] = 255;

        let (encoded, state) = codec.encode(&data, &config).unwrap();
        let decoded = codec.decode(&encoded, &state).unwrap();

        assert_eq!(data, decoded);

        // Skewed data should compress well
        assert!(
            encoded.len() < data.len(),
            "Skewed data should compress: {} >= {}",
            encoded.len(),
            data.len()
        );
    }

    #[test]
    fn test_disabled() {
        let codec = CpuEntropyCodec::new();
        let config = EntropyConfig { enabled: false };

        let data = vec![1u8, 2, 3, 4, 5];
        let (encoded, state) = codec.encode(&data, &config).unwrap();
        assert_eq!(encoded, data);

        let decoded = codec.decode(&encoded, &state).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_empty() {
        let codec = CpuEntropyCodec::new();
        let config = EntropyConfig { enabled: true };

        let data: Vec<u8> = vec![];
        let (encoded, state) = codec.encode(&data, &config).unwrap();
        let decoded = codec.decode(&encoded, &state).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_frequencies() {
        let data = vec![0u8, 0, 0, 1, 1, 2];
        let freq = compute_frequencies(&data);
        assert_eq!(freq[0], 3);
        assert_eq!(freq[1], 2);
        assert_eq!(freq[2], 1);
        assert_eq!(freq[3], 0);
    }
}
