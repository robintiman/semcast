//! Live end-to-end test against a decision model behind llama.cpp's
//! `llama-server`. Ignored by default:
//!
//! ```sh
//! llama-server -m span02-mxfp4.gguf --port 8090 -np 4 -c 16384 -b 4096 -ub 4096
//! ollama pull gemma4:e4b        # the session model; or export SEMCAST_OLLAMA_MODEL
//! cargo test --test live_systemone -- --ignored --nocapture
//! ```
//!
//! `SEMCAST_SYSTEMONE_URL` overrides `http://127.0.0.1:8090`. Against Ollama,
//! name the model too: `SEMCAST_SYSTEMONE_URL=http://127.0.0.1:11434
//! SEMCAST_SYSTEMONE_MODEL=d3`.

use std::sync::Arc;

use datafusion::arrow::array::{Float64Array, Int64Array};
use semcast::SemcastContextBuilder;
use semcast::model::{ModelProvider, OllamaProvider, SystemOneProvider};

#[tokio::test]
#[ignore = "requires a running server with a decision model"]
async fn means_and_confidence_against_a_live_decider() {
    let url = std::env::var("SEMCAST_SYSTEMONE_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8090".to_owned());
    let model = std::env::var("SEMCAST_OLLAMA_MODEL").unwrap_or_else(|_| "gemma4:e4b".to_owned());
    let named = std::env::var("SEMCAST_SYSTEMONE_MODEL").ok();
    let decider = SystemOneProvider::connect(&url, named.as_deref())
        .await
        .unwrap();
    println!("decider: {}", decider.id());
    // The session model is never asked here; MEANS belongs to the decider.
    let ctx = SemcastContextBuilder::new(Arc::new(OllamaProvider::new(model)))
        .with_decider(Arc::new(decider))
        .build();

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
        "SELECT meeting_id, CONFIDENCE(transcript MEANS 'a product launch is planned') AS p
         FROM meetings
         WHERE transcript MEANS 'a product launch is planned'",
    )
    .await
    .unwrap()
    .collect()
    .await
    .unwrap();

    let ids: Vec<i64> = batches
        .iter()
        .flat_map(|b| {
            let column = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            column.values().to_vec()
        })
        .collect();
    assert_eq!(ids, vec![1]);
    let p = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);
    println!("P(yes) = {p}");
    assert!(p > 0.5, "{p}");
}
