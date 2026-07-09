//! Semantic types — the whole specification of a typed extraction
//! (`CREATE SEMANTIC TYPE`): field names, types, and a one-line doc per
//! field. semcast synthesizes the prompt from this; users never write one.

pub mod registry;

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field};
use datafusion::error::DataFusionError;
use serde_json::{Value, json};

/// Version of the synthesized extraction prompt scheme. Participates in cache
/// keys: bump it and every cached extraction is honestly invalidated.
pub const EXTRACT_PROMPT_VERSION: &str = "extract-v1";

/// A named, versioned extraction spec. `version` participates in cache keys:
/// editing one field's doc line invalidates exactly that field's entries.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub struct SemanticType {
    pub name: String,
    pub version: String,
    pub fields: Vec<FieldSpec>,
    /// `TOGETHER(...)` groups, by field name. Members are co-generated in one
    /// shot; the planner never prunes one member without the others. Fields
    /// not listed here are independent — that's what enables field pushdown.
    pub together: Vec<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub struct FieldSpec {
    pub name: String,
    pub ty: FieldType,
    /// The one-line natural-language doc the prompt is synthesized from.
    pub doc: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub enum FieldType {
    /// Free-form string — the only truly "prose" field; stays with the model.
    Text,
    /// Aggregates in SQL (`avg`, `sum`) — no LLM at rollup.
    Int,
    Real,
    /// `REAL CHECK (a..b)` — validated at decode time.
    RealBounded {
        min: OrderedF64,
        max: OrderedF64,
    },
    /// Becomes a plain predicate.
    Bool,
    /// `ONEOF(a, b, c)` — closed category; `GROUP BY`-able.
    OneOf(Vec<String>),
    /// `LEVEL(a, b, c)` — ordered category, declared low→high; comparable.
    Level(Vec<String>),
    /// `T[]` — multi-valued extraction.
    List(Box<FieldType>),
    /// A nested semantic type, by name — compose structured extractions.
    Nested(String),
}

impl SemanticType {
    /// Synthesize the extraction prompt for a subset of fields (field pushdown
    /// hands us only the fields the query actually uses). Deterministic
    /// byte-for-byte and order-independent — the field subset alone decides
    /// the output, so it can key a cache and feed cost estimation.
    pub fn synthesize_prompt(&self, fields: &[&str]) -> crate::Result<String> {
        let specs = self.select(fields)?;
        let mut prompt = String::from(
            "You extract structured facts from a document. Return a single JSON \
             object with exactly these keys, and nothing else. Base every value \
             only on the document; if a fact is not present, use null.\n",
        );
        for spec in specs {
            prompt.push_str(&format!(
                "\n- {} ({}): {}",
                spec.name,
                spec.ty.describe(),
                spec.doc
            ));
        }
        Ok(prompt)
    }

    /// A JSON Schema for a single object covering `fields` — the constrained
    /// decoding contract. `additionalProperties: false` and every field
    /// `required` (both demanded by the hosted structured-output API); numeric
    /// bounds live in the field description, never as `minimum`/`maximum`
    /// (unsupported by the API subset and validated client-side regardless).
    pub fn json_schema(&self, fields: &[&str]) -> crate::Result<Value> {
        let specs = self.select(fields)?;
        let mut properties = serde_json::Map::new();
        let mut required = Vec::with_capacity(specs.len());
        for spec in &specs {
            let mut prop = spec.ty.json_type()?;
            let mut description = spec.doc.clone();
            if let FieldType::RealBounded { min, max } = &spec.ty {
                description.push_str(&format!(" (between {} and {})", min.0, max.0));
            }
            let object = prop
                .as_object_mut()
                .expect("json_type always returns an object");
            object.insert("description".to_owned(), json!(description));
            properties.insert(spec.name.clone(), prop);
            required.push(json!(spec.name));
        }
        Ok(json!({
            "type": "object",
            "additionalProperties": false,
            "required": required,
            "properties": properties,
        }))
    }

    /// The Arrow type a field decodes into. `ONEOF`/`LEVEL` land as plain
    /// `Utf8` today — dictionary encoding buys nothing for `GROUP BY`, and
    /// `LEVEL`'s ordered comparability needs rank semantics beyond an Arrow
    /// dictionary, so both are deferred.
    pub fn arrow_type(&self, field: &str) -> crate::Result<DataType> {
        self.field(field)?.ty.arrow_type()
    }

    /// The generation units, in declaration order: each `TOGETHER` group is
    /// one unit, and every ungrouped field is a singleton unit. A unit is the
    /// granularity of one model call and one cache entry — members of a group
    /// are always generated (and cached) together.
    pub fn generation_units(&self) -> Vec<Vec<&FieldSpec>> {
        let mut units: Vec<Vec<&FieldSpec>> = Vec::new();
        let mut seen: HashSet<&str> = HashSet::new();
        for spec in &self.fields {
            if seen.contains(spec.name.as_str()) {
                continue;
            }
            match self.group_of(&spec.name) {
                Some(group) => {
                    let unit: Vec<&FieldSpec> = self
                        .fields
                        .iter()
                        .filter(|f| group.iter().any(|m| m == &f.name))
                        .collect();
                    for member in &unit {
                        seen.insert(member.name.as_str());
                    }
                    units.push(unit);
                }
                None => {
                    seen.insert(spec.name.as_str());
                    units.push(vec![spec]);
                }
            }
        }
        units
    }

    /// Restrict the type to `fields` plus the `TOGETHER` closure — any group
    /// with a referenced member pulls in all its members, so co-generation
    /// stays intact. This *is* field pushdown: the returned type carries only
    /// the fields the query can reach. Declaration order and surviving groups
    /// are preserved.
    pub fn pruned(&self, fields: &[&str]) -> crate::Result<SemanticType> {
        let mut keep: HashSet<&str> = HashSet::new();
        for &name in fields {
            self.field(name)?;
            keep.insert(name);
        }
        for group in &self.together {
            if group.iter().any(|m| keep.contains(m.as_str())) {
                for member in group {
                    keep.insert(member.as_str());
                }
            }
        }
        let fields = self
            .fields
            .iter()
            .filter(|f| keep.contains(f.name.as_str()))
            .cloned()
            .collect();
        let together = self
            .together
            .iter()
            .filter(|g| g.iter().any(|m| keep.contains(m.as_str())))
            .cloned()
            .collect();
        Ok(SemanticType {
            name: self.name.clone(),
            version: self.version.clone(),
            fields,
            together,
        })
    }

    /// The declaration-ordered specs matching `fields`; errors on any unknown
    /// name. Order-independent so callers get a stable, deduplicated view.
    fn select(&self, fields: &[&str]) -> crate::Result<Vec<&FieldSpec>> {
        let want: HashSet<&str> = fields.iter().copied().collect();
        for &name in fields {
            self.field(name)?;
        }
        Ok(self
            .fields
            .iter()
            .filter(|f| want.contains(f.name.as_str()))
            .collect())
    }

    fn field(&self, name: &str) -> crate::Result<&FieldSpec> {
        self.fields
            .iter()
            .find(|f| f.name == name)
            .ok_or_else(|| plan_error(format!("semantic type {} has no field {name}", self.name)))
    }

    fn group_of(&self, name: &str) -> Option<&Vec<String>> {
        self.together.iter().find(|g| g.iter().any(|m| m == name))
    }
}

/// A stable hash of a generation unit's specs (name + type + doc of every
/// member). Editing one field's doc changes exactly its unit's hash, so the
/// cache invalidates that unit and leaves sibling units valid.
pub fn unit_hash(specs: &[&FieldSpec]) -> u64 {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    for spec in specs {
        spec.hash(&mut hasher);
    }
    hasher.finish()
}

impl FieldType {
    /// The Arrow type this field decodes into.
    pub fn arrow_type(&self) -> crate::Result<DataType> {
        Ok(match self {
            FieldType::Text | FieldType::OneOf(_) | FieldType::Level(_) => DataType::Utf8,
            FieldType::Int => DataType::Int64,
            FieldType::Real | FieldType::RealBounded { .. } => DataType::Float64,
            FieldType::Bool => DataType::Boolean,
            FieldType::List(inner) => {
                DataType::List(Arc::new(Field::new("item", inner.arrow_type()?, true)))
            }
            FieldType::Nested(name) => {
                return Err(not_implemented(format!(
                    "nested semantic type {name} is not implemented yet"
                )));
            }
        })
    }

    /// The type portion of this field's JSON Schema (no description).
    fn json_type(&self) -> crate::Result<Value> {
        Ok(match self {
            FieldType::Text => json!({"type": "string"}),
            FieldType::Int => json!({"type": "integer"}),
            FieldType::Real | FieldType::RealBounded { .. } => json!({"type": "number"}),
            FieldType::Bool => json!({"type": "boolean"}),
            FieldType::OneOf(variants) | FieldType::Level(variants) => {
                json!({"type": "string", "enum": variants})
            }
            FieldType::List(inner) => json!({"type": "array", "items": inner.json_type()?}),
            FieldType::Nested(name) => {
                return Err(not_implemented(format!(
                    "nested semantic type {name} is not implemented yet"
                )));
            }
        })
    }

    /// Natural-language description for the synthesized prompt.
    fn describe(&self) -> String {
        match self {
            FieldType::Text => "free-form text".to_owned(),
            FieldType::Int => "an integer".to_owned(),
            FieldType::Real => "a number".to_owned(),
            FieldType::RealBounded { min, max } => {
                format!("a number between {} and {}", min.0, max.0)
            }
            FieldType::Bool => "true or false".to_owned(),
            FieldType::OneOf(variants) => format!("one of: {}", variants.join(", ")),
            FieldType::Level(variants) => {
                format!(
                    "one of these levels, ordered low to high: {}",
                    variants.join(", ")
                )
            }
            FieldType::List(inner) => format!("a list where each item is {}", inner.describe()),
            FieldType::Nested(name) => format!("a nested {name} object"),
        }
    }
}

/// Canonical DDL rendering — round-trips with the field-type parser, and feeds
/// both version hashing and the inline-`EXTRACT` typespec literal.
impl fmt::Display for FieldType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FieldType::Text => write!(f, "TEXT"),
            FieldType::Int => write!(f, "INT"),
            FieldType::Real => write!(f, "REAL"),
            FieldType::RealBounded { min, max } => write!(f, "REAL CHECK ({}..{})", min.0, max.0),
            FieldType::Bool => write!(f, "BOOL"),
            FieldType::OneOf(variants) => write!(f, "ONEOF({})", variants.join(",")),
            FieldType::Level(variants) => write!(f, "LEVEL({})", variants.join(",")),
            FieldType::List(inner) => write!(f, "{inner}[]"),
            FieldType::Nested(name) => write!(f, "{name}"),
        }
    }
}

fn plan_error(message: String) -> crate::SemcastError {
    crate::SemcastError::DataFusion(DataFusionError::Plan(message))
}

fn not_implemented(message: String) -> crate::SemcastError {
    crate::SemcastError::DataFusion(DataFusionError::NotImplemented(message))
}

/// `f64` with total equality/ordering via bit patterns, so field types can be
/// hashed and compared inside logical plan nodes.
#[derive(Debug, Clone, Copy)]
pub struct OrderedF64(pub f64);

impl PartialEq for OrderedF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for OrderedF64 {}

impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.0.total_cmp(&other.0))
    }
}

impl std::hash::Hash for OrderedF64 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(name: &str, ty: FieldType, doc: &str) -> FieldSpec {
        FieldSpec {
            name: name.to_owned(),
            ty,
            doc: doc.to_owned(),
        }
    }

    /// The README's `MeetingFacts`: two independent `TEXT[]` fields and a
    /// `TOGETHER` group of `launch_stage ONEOF(...)` + `stage_quote TEXT`.
    fn meeting_facts() -> SemanticType {
        SemanticType {
            name: "MeetingFacts".to_owned(),
            version: "v1".to_owned(),
            fields: vec![
                field(
                    "products",
                    FieldType::List(Box::new(FieldType::Text)),
                    "product names discussed in this meeting",
                ),
                field(
                    "decisions",
                    FieldType::List(Box::new(FieldType::Text)),
                    "concrete decisions that were made",
                ),
                field(
                    "launch_stage",
                    FieldType::OneOf(vec![
                        "none".to_owned(),
                        "idea".to_owned(),
                        "planned".to_owned(),
                        "scheduled".to_owned(),
                        "shipped".to_owned(),
                    ]),
                    "the furthest launch stage discussed",
                ),
                field(
                    "stage_quote",
                    FieldType::Text,
                    "the transcript line that shows that stage",
                ),
            ],
            together: vec![vec!["launch_stage".to_owned(), "stage_quote".to_owned()]],
        }
    }

    #[test]
    fn arrow_types_map_per_field() {
        let ty = meeting_facts();
        assert_eq!(
            ty.arrow_type("launch_stage").unwrap(),
            DataType::Utf8,
            "ONEOF is plain Utf8",
        );
        assert_eq!(ty.arrow_type("stage_quote").unwrap(), DataType::Utf8);
        assert_eq!(
            ty.arrow_type("products").unwrap(),
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
        );
    }

    #[test]
    fn scalar_arrow_types() {
        assert_eq!(FieldType::Int.arrow_type().unwrap(), DataType::Int64);
        assert_eq!(FieldType::Real.arrow_type().unwrap(), DataType::Float64);
        assert_eq!(
            FieldType::RealBounded {
                min: OrderedF64(0.0),
                max: OrderedF64(1.0),
            }
            .arrow_type()
            .unwrap(),
            DataType::Float64,
        );
        assert_eq!(FieldType::Bool.arrow_type().unwrap(), DataType::Boolean);
        assert_eq!(
            FieldType::Level(vec!["low".to_owned(), "high".to_owned()])
                .arrow_type()
                .unwrap(),
            DataType::Utf8,
        );
    }

    #[test]
    fn unknown_field_is_a_plan_error() {
        let err = meeting_facts().arrow_type("nonesuch").unwrap_err();
        assert!(err.to_string().contains("no field nonesuch"), "got: {err}");
    }

    #[test]
    fn nested_arrow_type_is_deferred() {
        let err = FieldType::Nested("Other".to_owned())
            .arrow_type()
            .unwrap_err();
        assert!(err.to_string().contains("not implemented"), "got: {err}");
    }

    #[test]
    fn prompt_is_order_independent_and_lists_requested_fields() {
        let ty = meeting_facts();
        let a = ty.synthesize_prompt(&["launch_stage", "products"]).unwrap();
        let b = ty.synthesize_prompt(&["products", "launch_stage"]).unwrap();
        assert_eq!(a, b, "the field subset alone decides the prompt");
        // Declaration order inside the prompt: products before launch_stage.
        assert!(a.find("products").unwrap() < a.find("launch_stage").unwrap());
        assert!(a.contains("one of: none, idea, planned, scheduled, shipped"));
        assert!(!a.contains("stage_quote"), "unrequested field is absent");
    }

    #[test]
    fn json_schema_shape() {
        let ty = meeting_facts();
        let schema = ty.json_schema(&["launch_stage", "products"]).unwrap();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["additionalProperties"], false);
        let required: Vec<&str> = schema["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(required.contains(&"launch_stage") && required.contains(&"products"));
        let props = schema["properties"].as_object().unwrap();
        assert_eq!(props["launch_stage"]["type"], "string");
        assert_eq!(
            props["launch_stage"]["enum"],
            json!(["none", "idea", "planned", "scheduled", "shipped"]),
        );
        assert_eq!(props["products"]["type"], "array");
        assert_eq!(props["products"]["items"]["type"], "string");
        assert!(props["launch_stage"]["description"].is_string());
    }

    #[test]
    fn json_schema_puts_real_bounds_in_the_description_only() {
        let ty = SemanticType {
            name: "T".to_owned(),
            version: "v1".to_owned(),
            fields: vec![field(
                "score",
                FieldType::RealBounded {
                    min: OrderedF64(0.0),
                    max: OrderedF64(1.0),
                },
                "confidence",
            )],
            together: vec![],
        };
        let schema = ty.json_schema(&["score"]).unwrap();
        let prop = &schema["properties"]["score"];
        assert_eq!(prop["type"], "number");
        assert!(prop.get("minimum").is_none() && prop.get("maximum").is_none());
        assert!(
            prop["description"]
                .as_str()
                .unwrap()
                .contains("between 0 and 1")
        );
    }

    #[test]
    fn generation_units_group_together_and_keep_declaration_order() {
        let ty = meeting_facts();
        let units = ty.generation_units();
        let names: Vec<Vec<&str>> = units
            .iter()
            .map(|u| u.iter().map(|f| f.name.as_str()).collect())
            .collect();
        assert_eq!(
            names,
            vec![
                vec!["products"],
                vec!["decisions"],
                vec!["launch_stage", "stage_quote"],
            ],
        );
    }

    #[test]
    fn pruning_expands_the_together_closure() {
        let ty = meeting_facts();
        // launch_stage pulls in its TOGETHER sibling stage_quote.
        let pruned = ty.pruned(&["launch_stage"]).unwrap();
        let names: Vec<&str> = pruned.fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, vec!["launch_stage", "stage_quote"]);
        assert_eq!(
            pruned.together,
            vec![vec!["launch_stage".to_owned(), "stage_quote".to_owned()]]
        );

        // An independent field prunes to just itself, no groups.
        let pruned = ty.pruned(&["products"]).unwrap();
        let names: Vec<&str> = pruned.fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, vec!["products"]);
        assert!(pruned.together.is_empty());
    }

    #[test]
    fn pruning_unknown_field_errors() {
        assert!(meeting_facts().pruned(&["nope"]).is_err());
    }

    #[test]
    fn unit_hash_tracks_doc_edits() {
        let ty = meeting_facts();
        let stage_quote = ty.field("stage_quote").unwrap();
        let launch_stage = ty.field("launch_stage").unwrap();
        let base = unit_hash(&[launch_stage, stage_quote]);

        let mut edited = ty.clone();
        edited.fields[3].doc = "a different doc line".to_owned();
        let edited_quote = &edited.fields[3];
        let edited_stage = &edited.fields[2];
        assert_ne!(base, unit_hash(&[edited_stage, edited_quote]));
    }

    #[test]
    fn field_type_display_round_trip_shapes() {
        assert_eq!(FieldType::Text.to_string(), "TEXT");
        assert_eq!(FieldType::Int.to_string(), "INT");
        assert_eq!(FieldType::Bool.to_string(), "BOOL");
        assert_eq!(FieldType::Real.to_string(), "REAL");
        assert_eq!(
            FieldType::RealBounded {
                min: OrderedF64(0.0),
                max: OrderedF64(1.0),
            }
            .to_string(),
            "REAL CHECK (0..1)",
        );
        assert_eq!(
            FieldType::OneOf(vec!["none".to_owned(), "idea".to_owned()]).to_string(),
            "ONEOF(none,idea)",
        );
        assert_eq!(
            FieldType::Level(vec!["low".to_owned(), "high".to_owned()]).to_string(),
            "LEVEL(low,high)",
        );
        assert_eq!(
            FieldType::List(Box::new(FieldType::Text)).to_string(),
            "TEXT[]",
        );
    }
}
