//! Live tests against the Voyage AI embeddings API. Ignored by default:
//!
//! ```sh
//! export VOYAGE_API_KEY=...
//! cargo test --test live_voyage -- --ignored --nocapture
//! ```

use semcast::model::{ModelProvider, VoyageProvider};

#[tokio::test]
#[ignore = "requires VOYAGE_API_KEY"]
async fn embeds_texts_against_live_voyage() {
    let provider = VoyageProvider::from_env().expect("export VOYAGE_API_KEY to run this test");

    let embeddings = provider
        .embed(vec![
            "we agreed to ship offline sync in the third quarter".to_owned(),
            "status round about the cafeteria menu, nothing else".to_owned(),
        ])
        .await
        .unwrap();

    assert_eq!(embeddings.len(), 2, "one embedding per input");
    let dim = embeddings[0].len();
    println!("live voyage embedding dimension: {dim}");
    assert!(dim > 0);
    assert_eq!(embeddings[1].len(), dim, "consistent dimension");
    assert_ne!(
        embeddings[0], embeddings[1],
        "distinct texts, distinct vectors"
    );
}
