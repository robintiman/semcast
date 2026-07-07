//! Live end-to-end test against a local Ollama server. Ignored by default:
//!
//! ```sh
//! ollama pull gemma4:31b   # or export SEMCAST_OLLAMA_MODEL=<model>
//! cargo test --test live_ollama -- --ignored --nocapture
//! ```

use std::sync::Arc;

use semcast::model::OllamaProvider;
use semcast::semcast_context;

#[tokio::test]
#[ignore = "requires a running Ollama server with a pulled model"]
async fn means_filter_against_live_ollama() {
    let model = std::env::var("SEMCAST_OLLAMA_MODEL").unwrap_or_else(|_| "gemma4:31b".to_owned());
    let ctx = semcast_context(Arc::new(OllamaProvider::new(model)));

    ctx.sql(
        "CREATE TABLE meetings AS
         SELECT * FROM (VALUES
             (1, 'we agreed to ship offline sync in the third quarter'),
             (2, 'status round about the cafeteria menu, nothing else')
         ) AS t(meeting_id, transcript)",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let batches = semcast::sql(
        &ctx,
        "SELECT meeting_id FROM meetings
         WHERE transcript MEANS 'discussed shipping an offline sync feature'
         ORDER BY meeting_id",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    println!("live ollama verify kept {rows} of 2 rows");
    // Meeting 1 unambiguously matches; any reasonable model keeps it and
    // drops the cafeteria meeting.
    assert_eq!(
        rows, 1,
        "expected exactly the offline-sync meeting to survive"
    );
}
