//! A decision model behind the System One API (`POST /v1/systemone`), as
//! served by llama.cpp's `llama-server` and by Ollama — the session's
//! optional *decider*.
//!
//! A decision model does not write: it reads a state and a typed question
//! and returns a probability per answer. That fits `MEANS` exactly (one
//! yes/no question per row, P("yes") for free) and nothing generative, so
//! this provider answers requests that carry a [`YesNoAsk`] and refuses the
//! rest; the planner sends it only `MEANS` questions.
//!
//! The question is worded here, not taken from `request.system`: that prompt
//! tells a generative model to answer "yes or no", while a decision model
//! picks an option code from its own template. The wording is part of the
//! [`ModelId`] ([`ASK_VERSION`]), so changing it invalidates this provider's
//! verdicts and no one else's.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::{Completion, CompletionRequest, Embedding, ModelId, ModelProvider, YesNoAsk};
use crate::{Result, SemcastError};

/// Requests in flight to a decider by default: a 26B decision model serves
/// about this many slots in 36 GB.
pub const DEFAULT_DECIDER_MAX_IN_FLIGHT: usize = 4;

/// Version of [`instructions`]. Bump it when the wording changes.
const ASK_VERSION: &str = "ask-v1";

#[derive(Debug, Clone)]
pub struct SystemOneProvider {
    base_url: String,
    /// Names the answers in cache keys.
    model: String,
    /// The model as the server names it: sent with each question, and
    /// expected back on each answer.
    served: String,
    client: reqwest::Client,
}

impl SystemOneProvider {
    /// A provider for the decision model `model` served at `base_url`.
    /// `model` names the answers in cache keys and is sent with each
    /// question; prefer [`connect`](Self::connect), which checks the server.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        let mut base_url = base_url.into();
        base_url.truncate(base_url.trim_end_matches('/').len());
        let model = model.into();
        Self {
            base_url,
            served: model.clone(),
            model,
            client: reqwest::Client::new(),
        }
    }

    /// Connect to the server at `base_url` (`GET /v1/models`). With no
    /// `model`, the provider is named after the one model the server lists
    /// (its file name, if it is a path) — `llama-server` — so cached
    /// verdicts follow the model that gave them. A server with several
    /// (Ollama) needs `model` named, and must list it. Fails if the server
    /// is unreachable, serves no such model, or leaves the choice open.
    pub async fn connect(base_url: impl Into<String>, model: Option<&str>) -> Result<Self> {
        let mut provider = Self::new(base_url, String::new());
        let response = super::send_with_retry("systemone", || {
            provider
                .client
                .get(format!("{}/v1/models", provider.base_url))
        })
        .await?;
        let models: Models = response
            .json()
            .await
            .map_err(|e| SemcastError::Model(format!("invalid systemone model list: {e}")))?;
        let served: Vec<String> = models.data.into_iter().map(|model| model.id).collect();
        // `llama-server` lists the path it was started with; the file name
        // alone keeps verdicts across a restart from elsewhere.
        let file = |id: &str| id.rsplit_once('/').map_or(id, |(_, file)| file).to_owned();
        (provider.model, provider.served) = match (model, served.as_slice()) {
            // Ollama lists `d3` as `d3:latest`, and takes either name.
            (Some(name), _) => {
                if served
                    .iter()
                    .any(|id| id == name || id.strip_suffix(":latest") == Some(name))
                {
                    (name.to_owned(), name.to_owned())
                } else if let Some(id) = served.iter().find(|id| file(id) == name) {
                    (name.to_owned(), id.clone())
                } else {
                    return Err(SemcastError::Model(format!(
                        "{} does not serve a model named {name}",
                        provider.base_url
                    )));
                }
            }
            // Named as `--decider-model` would name it, so adding the flag
            // later keeps the stored verdicts.
            (None, [id]) if !id.is_empty() => {
                let name = file(id);
                let name = name.strip_suffix(":latest").unwrap_or(&name).to_owned();
                (name, id.clone())
            }
            (None, [] | [_]) => {
                return Err(SemcastError::Model(format!(
                    "{} serves no model",
                    provider.base_url
                )));
            }
            (None, _) => {
                return Err(SemcastError::Model(format!(
                    "{} serves {} models; name the decision model",
                    provider.base_url,
                    served.len()
                )));
            }
        };
        Ok(provider)
    }

    async fn complete_one(&self, request: CompletionRequest) -> Result<Completion> {
        let Some(ask) = &request.ask else {
            return Err(SemcastError::Model(
                "a decision model answers yes/no questions only".to_owned(),
            ));
        };
        let body = Request {
            model: &self.served,
            state: &request.input,
            questions: Questions {
                q: Question {
                    kind: "noul",
                    instructions: instructions(ask),
                },
            },
        };
        tracing::debug!(
            target: "semcast::model",
            model = %self.model,
            instructions = %body.questions.q.instructions,
            input = %request.input,
            "decision request"
        );
        let response = super::send_with_retry("systemone", || {
            self.client
                .post(format!("{}/v1/systemone", self.base_url))
                .json(&body)
        })
        .await?;
        let answered: Response = response
            .json()
            .await
            .map_err(|e| SemcastError::Model(format!("invalid systemone response: {e}")))?;
        // A server restarted with another model keeps answering; its
        // verdicts must not be stored under this one's name.
        if let Some(model) = &answered.model
            && *model != self.served
            && model.strip_suffix(":latest") != Some(&self.served)
        {
            return Err(SemcastError::Model(format!(
                "systemone answered with {model}, not {}",
                self.served
            )));
        }
        let yes = answered.answers.q.noul;
        if !(0.0..=1.0).contains(&yes) {
            return Err(SemcastError::Model(format!(
                "systemone returned P(yes) = {yes}"
            )));
        }
        tracing::debug!(
            target: "semcast::model",
            model = %self.model,
            yes_probability = yes,
            "decision response"
        );
        Ok(Completion {
            text: if yes >= 0.5 { "yes" } else { "no" }.to_owned(),
            input_tokens: answered.usage.input_tokens,
            output_tokens: answered.usage.output_tokens,
            yes_probability: Some(yes),
            merged: false,
        })
    }
}

/// The question a decision model is asked about the state. Users never
/// write this. The excerpts wording is the one measured on CUAD.
fn instructions(ask: &YesNoAsk) -> String {
    let condition = &ask.condition;
    if ask.excerpts {
        format!(
            "The state holds excerpts from a document, separated by `---`. \
             The excerpts may be partial. \
             Do the excerpts show the document satisfies this predicate: {condition}?"
        )
    } else {
        format!(
            "The state holds a document. \
             Does the document satisfy this predicate: {condition}?"
        )
    }
}

#[async_trait]
impl ModelProvider for SystemOneProvider {
    fn id(&self) -> ModelId {
        ModelId(format!("systemone/{}/{ASK_VERSION}", self.model))
    }

    async fn complete(&self, requests: Vec<CompletionRequest>) -> Vec<Result<Completion>> {
        super::complete_concurrently(requests, |req| self.complete_one(req)).await
    }

    async fn embed(&self, _texts: Vec<String>) -> Result<Vec<Embedding>> {
        Err(SemcastError::Model(
            "a decision model cannot embed".to_owned(),
        ))
    }
}

#[derive(Serialize)]
struct Request<'a> {
    /// Ollama picks the model by this; `llama-server` serves its one model.
    model: &'a str,
    state: &'a str,
    questions: Questions,
}

/// One question per request, under a fixed key.
#[derive(Serialize)]
struct Questions {
    q: Question,
}

#[derive(Serialize)]
struct Question {
    #[serde(rename = "type")]
    kind: &'static str,
    instructions: String,
}

#[derive(Deserialize)]
struct Response {
    #[serde(default)]
    model: Option<String>,
    answers: Answers,
    #[serde(default)]
    usage: Usage,
}

#[derive(Deserialize)]
struct Answers {
    q: Answer,
}

#[derive(Deserialize)]
struct Answer {
    /// P("yes").
    noul: f32,
}

#[derive(Deserialize, Default)]
struct Usage {
    #[serde(default)]
    input_tokens: usize,
    #[serde(default)]
    output_tokens: usize,
}

#[derive(Deserialize)]
struct Models {
    data: Vec<Model>,
}

#[derive(Deserialize)]
struct Model {
    id: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ask(excerpts: bool) -> YesNoAsk {
        YesNoAsk {
            condition: "a refund is requested".to_owned(),
            excerpts,
        }
    }

    #[test]
    fn the_question_is_one_noul_about_the_state() {
        let body = serde_json::to_value(Request {
            model: "span-02",
            state: "I want my money back",
            questions: Questions {
                q: Question {
                    kind: "noul",
                    instructions: instructions(&ask(false)),
                },
            },
        })
        .unwrap();
        assert_eq!(body["model"], "span-02");
        assert_eq!(body["state"], "I want my money back");
        assert_eq!(body["questions"]["q"]["type"], "noul");
        assert_eq!(
            body["questions"]["q"]["instructions"],
            "The state holds a document. \
             Does the document satisfy this predicate: a refund is requested?"
        );
    }

    #[test]
    fn excerpts_are_asked_about_as_excerpts() {
        // The wording measured on CUAD; changing it means bumping ASK_VERSION.
        assert_eq!(
            instructions(&ask(true)),
            "The state holds excerpts from a document, separated by `---`. \
             The excerpts may be partial. \
             Do the excerpts show the document satisfies this predicate: a refund is requested?"
        );
    }

    #[test]
    fn a_live_answer_decodes() {
        // Shape of a llama-server /v1/systemone reply.
        let reply = r#"{"model":"span02-mxfp4.gguf",
            "answers":{"q":{"type":"noul","noul":0.9998662}},
            "usage":{"input_tokens":251,"output_tokens":0}}"#;
        let answered: Response = serde_json::from_str(reply).unwrap();
        assert!(answered.answers.q.noul > 0.99);
        assert_eq!(answered.usage.input_tokens, 251);
    }

    #[test]
    fn the_id_names_the_model_and_the_wording() {
        let provider = SystemOneProvider::new("http://localhost:8090/", "span-02");
        assert_eq!(provider.id().0, "systemone/span-02/ask-v1");
        assert_eq!(provider.base_url, "http://localhost:8090");
    }

    #[tokio::test]
    async fn anything_but_a_yes_no_question_is_refused() {
        let provider = SystemOneProvider::new("http://127.0.0.1:1", "span-02");
        let results = provider
            .complete(vec![CompletionRequest {
                system: "Extract the fields.".to_owned(),
                input: "text".to_owned(),
                max_tokens: 64,
                schema: None,
                logprobs: false,
                ask: None,
            }])
            .await;
        assert!(results[0].is_err());
        assert!(provider.embed(vec!["text".to_owned()]).await.is_err());
    }
}
