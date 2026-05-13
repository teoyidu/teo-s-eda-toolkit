use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

/// Find duplicate lines using TF-IDF cosine similarity.
/// Returns a list of indices of lines to be removed.
#[pyfunction]
fn find_duplicate_indices_tfidf(lines: Vec<String>, similarity_threshold: f64) -> Vec<usize> {
    let n = lines.len();
    if n <= 1 {
        return vec![];
    }

    let mut doc_tokens: Vec<Vec<String>> = Vec::with_capacity(n);
    let mut df: HashMap<String, usize> = HashMap::new();

    for line in &lines {
        let words: Vec<&str> = line.split_whitespace().collect();
        let mut tokens = Vec::new();
        let mut seen = HashSet::new();
        
        // Unigrams
        for w in &words {
            let t = w.to_string();
            tokens.push(t.clone());
            if seen.insert(t.clone()) {
                *df.entry(t).or_insert(0) += 1;
            }
        }
        
        // Bigrams
        if words.len() > 1 {
            for i in 0..words.len() - 1 {
                let t = format!("{} {}", words[i], words[i+1]);
                tokens.push(t.clone());
                if seen.insert(t.clone()) {
                    *df.entry(t).or_insert(0) += 1;
                }
            }
        }
        
        doc_tokens.push(tokens);
    }

    // To match scikit-learn TfidfVectorizer max_features=1000
    let mut term_counts: HashMap<String, usize> = HashMap::new();
    for tokens in &doc_tokens {
        for t in tokens {
            *term_counts.entry(t.clone()).or_insert(0) += 1;
        }
    }

    let mut terms: Vec<(String, usize)> = term_counts.into_iter().collect();
    // Sort by count descending, then by string to be deterministic
    terms.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    
    // Take top 1000 features
    if terms.len() > 1000 {
        terms.truncate(1000);
    }

    let mut term_to_id: HashMap<String, usize> = HashMap::new();
    let mut idf: Vec<f64> = Vec::with_capacity(terms.len());
    let n_f64 = n as f64;

    for (i, (term, _)) in terms.iter().enumerate() {
        term_to_id.insert(term.clone(), i);
        let doc_freq = *df.get(term).unwrap_or(&0) as f64;
        // Scikit-learn smoothed IDF: ln((1 + N) / (1 + df)) + 1
        let val = ((1.0 + n_f64) / (1.0 + doc_freq)).ln() + 1.0;
        idf.push(val);
    }

    // Compute TF-IDF sparse vectors (L2 normalized)
    let mut doc_vectors: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);

    for tokens in &doc_tokens {
        let mut tf: HashMap<usize, f64> = HashMap::new();
        for t in tokens {
            if let Some(&id) = term_to_id.get(t) {
                *tf.entry(id).or_insert(0.0) += 1.0;
            }
        }
        
        let mut vec: Vec<(usize, f64)> = Vec::with_capacity(tf.len());
        let mut norm_sq = 0.0;
        for (id, count) in tf {
            let tfidf = count * idf[id];
            vec.push((id, tfidf));
            norm_sq += tfidf * tfidf;
        }
        
        // L2 normalize
        if norm_sq > 0.0 {
            let norm = norm_sq.sqrt();
            for item in &mut vec {
                item.1 /= norm;
            }
        }
        
        // Sort by id to make dot product linear
        vec.sort_by_key(|k| k.0);
        doc_vectors.push(vec);
    }

    let mut duplicates = HashSet::new();

    // Compute pairwise cosine similarity
    for i in 0..n {
        for j in (i + 1)..n {
            let vec_i = &doc_vectors[i];
            let vec_j = &doc_vectors[j];
            
            // Dot product of sorted sparse vectors
            let mut dot_product = 0.0;
            let mut p_i = 0;
            let mut p_j = 0;
            
            while p_i < vec_i.len() && p_j < vec_j.len() {
                if vec_i[p_i].0 == vec_j[p_j].0 {
                    dot_product += vec_i[p_i].1 * vec_j[p_j].1;
                    p_i += 1;
                    p_j += 1;
                } else if vec_i[p_i].0 < vec_j[p_j].0 {
                    p_i += 1;
                } else {
                    p_j += 1;
                }
            }
            
            if dot_product > similarity_threshold {
                duplicates.insert(j);
            }
        }
    }

    let mut result: Vec<usize> = duplicates.into_iter().collect();
    result.sort_unstable();
    result
}

#[pymodule]
fn rust_ext(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(find_duplicate_indices_tfidf, m)?)?;
    Ok(())
}
