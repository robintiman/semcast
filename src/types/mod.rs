//! Semantic types — the whole specification of a typed extraction
//! (`CREATE SEMANTIC TYPE`): field names, types, and a one-line doc per
//! field. semcast synthesizes the prompt from this; users never write one.

use datafusion::arrow::datatypes::DataType;

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
    RealBounded { min: OrderedF64, max: OrderedF64 },
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
    /// hands us only the fields the query actually uses).
    pub fn synthesize_prompt(&self, _fields: &[&str]) -> crate::Result<String> {
        todo!("prompt synthesis from field specs (with typed extraction)")
    }

    /// The Arrow type each field decodes into (`ONEOF` → dictionary,
    /// `LEVEL` → ordered dictionary, `T[]` → list, ...).
    pub fn arrow_type(&self, _field: &str) -> crate::Result<DataType> {
        todo!("FieldType → Arrow DataType mapping (with typed extraction)")
    }
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
